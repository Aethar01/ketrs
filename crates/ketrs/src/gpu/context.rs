//! The Vulkan device, compute pipelines and buffers.

use crate::error::Error;
use ash::vk;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

fn gpu_err(what: &str) -> impl Fn(vk::Result) -> Error + '_ {
    move |e| Error::Gpu(format!("{what}: {e}"))
}

/// The compute kernels, in pipeline order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    Pair = 0,
    Quad = 1,
    Pauli = 2,
    Phase = 3,
    /// One group of Pauli terms of an expectation value (`expect.comp`).
    Expect = 4,
    /// Probability totals of blocks of amplitudes (`probs.comp`).
    Probs = 5,
    /// `⟨λ|G|ψ⟩` against the partner state, for gradients (`overlap.comp`).
    Overlap = 6,
    /// `H|ψ⟩` into the partner state (`hamil.comp`).
    Hamil = 7,
    /// partner += a·ψ (`axpy.comp`).
    Axpy = 8,
    /// A run of gates on a few qubits in shared memory (`tile.comp`).
    Tile = 9,
}

impl Kernel {
    /// Workgroups to dispatch for `units` units of work. Tiles and blocks
    /// take one workgroup each; reductions are capped so their partial sums
    /// stay small; other kernels use one thread per unit.
    pub(crate) fn groups(self, units: u32) -> u32 {
        let groups = match self {
            Kernel::Tile | Kernel::Probs => units,
            Kernel::Expect | Kernel::Overlap => units.div_ceil(WORKGROUP).min(REDUCTION_GROUPS),
            _ => units.div_ceil(WORKGROUP),
        };
        groups.clamp(1, MAX_GROUPS)
    }
}

/// SPIR-V compiled by `build.rs` into `OUT_DIR`.
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Each kernel before `Tile`, as [single-buffer, chunked] variants.
const SHADERS: [[&[u8]; 2]; 9] = [
    [spirv!("pair_0"), spirv!("pair_1")],
    [spirv!("quad_0"), spirv!("quad_1")],
    [spirv!("pauli_0"), spirv!("pauli_1")],
    [spirv!("phase_0"), spirv!("phase_1")],
    [spirv!("expect_0"), spirv!("expect_1")],
    [spirv!("probs_0"), spirv!("probs_1")],
    [spirv!("overlap_0"), spirv!("overlap_1")],
    [spirv!("hamil_0"), spirv!("hamil_1")],
    [spirv!("axpy_0"), spirv!("axpy_1")],
];

/// The tile kernel for 2^bits-amplitude tiles, by bits, as [single-buffer,
/// chunked] variants.
const TILE_SHADERS: [(u32, [&[u8]; 2]); 3] = [
    (10, [spirv!("tile_0_10"), spirv!("tile_1_10")]),
    (11, [spirv!("tile_0_11"), spirv!("tile_1_11")]),
    (12, [spirv!("tile_0_12"), spirv!("tile_1_12")]),
];

/// State buffers a chunked kernel can address (`MAX_CHUNKS` in
/// `shaders/common.glsl`).
pub(crate) const MAX_CHUNKS: usize = 16;

/// Workgroups in an expectation-value dispatch, so each one writes at most
/// this many partial sums.
pub(crate) const REDUCTION_GROUPS: u32 = 1024;

/// Largest tile to use when the device allows it. Smaller tiles mean more
/// workgroups resident at once; on an RX 9060 XT, 10-qubit tiles beat 11
/// and 12 by 10–50%, and 10 fits Vulkan's guaranteed 16 KiB of shared
/// memory, so every device can use it.
const PREFERRED_TILE_BITS: u32 = 10;

/// Threads per workgroup, as declared in `shaders/common.glsl`.
pub(crate) const WORKGROUP: u32 = 256;
/// Workgroups per dispatch; shaders loop over any remaining work.
pub(crate) const MAX_GROUPS: u32 = 65_535;

/// Push constants shared by every kernel (see `shaders/common.glsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Push {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub units: u32,
    pub offset: u32,
    /// Amplitudes per state buffer, as a power of two (chunked kernels).
    pub chunk_bits: u32,
}

impl Push {
    fn bytes(&self) -> [u8; 24] {
        let words = [
            self.a,
            self.b,
            self.c,
            self.units,
            self.offset,
            self.chunk_bits,
        ];
        let mut out = [0u8; 24];
        for (chunk, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = w.to_ne_bytes();
        }
        out
    }
}

/// Destroys the instance unless taken, so early errors don't leak it.
struct InstanceGuard(Option<ash::Instance>);

impl InstanceGuard {
    fn get(&self) -> &ash::Instance {
        self.0.as_ref().expect("instance present until taken")
    }
    fn take(mut self) -> ash::Instance {
        self.0.take().expect("instance present until taken")
    }
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        if let Some(instance) = self.0.take() {
            // SAFETY: nothing created from the instance is still alive.
            unsafe { instance.destroy_instance(None) };
        }
    }
}

/// A Vulkan compute device able to run ketrs's statevector kernels.
///
/// Creating one loads the system Vulkan driver and picks a GPU with 64-bit
/// float support, preferring a discrete card. Share it through the returned
/// `Arc` across any number of [`GpuStatevector`](super::GpuStatevector)s.
pub struct GpuContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    memory: vk::PhysicalDeviceMemoryProperties,
    max_storage_range: u64,
    /// Whether kernels can index several state buffers, for states larger
    /// than one buffer (Vulkan 1.2 non-uniform descriptor indexing).
    chunks_supported: bool,
    /// Qubits per shared-memory tile.
    tile_bits: u32,
    name: String,
    command_pool: vk::CommandPool,
    pub(crate) descriptor_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    /// Serialises the command pool, descriptor pool and queue, which Vulkan
    /// requires to be externally synchronised.
    pub(crate) lock: Mutex<()>,
}

// SAFETY: Vulkan handles are plain identifiers; every use that Vulkan
// requires to be externally synchronised happens under `lock`.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Load Vulkan and open the best device with 64-bit float support.
    pub fn new() -> Result<Arc<Self>, Error> {
        // SAFETY: loads the system Vulkan library; failure is reported.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| Error::Gpu(format!("no Vulkan driver found: {e}")))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"ketrs")
            .api_version(vk::API_VERSION_1_2);
        // SAFETY: valid create info; the guard destroys the instance on error.
        let instance = InstanceGuard(Some(
            unsafe {
                entry.create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
            }
            .map_err(gpu_err("creating Vulkan instance"))?,
        ));

        let (physical, family) = Self::pick_device(instance.get())?;
        // SAFETY: plain queries on a valid physical device.
        let (props, memory) = unsafe {
            (
                instance.get().get_physical_device_properties(physical),
                instance
                    .get()
                    .get_physical_device_memory_properties(physical),
            )
        };
        // SAFETY: device_name is a NUL-terminated array filled by the driver.
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default().shader_float64(true);
        let chunks_supported = {
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut all = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
            // SAFETY: plain query on a Vulkan 1.2 device.
            unsafe {
                instance
                    .get()
                    .get_physical_device_features2(physical, &mut all)
            };
            v12.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE
        };
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_storage_buffer_array_non_uniform_indexing(chunks_supported);
        // SAFETY: the chosen device supports these features and the family.
        let device = unsafe {
            instance.get().create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_features(&features)
                    .push_next(&mut v12),
                None,
            )
        }
        .map_err(gpu_err("creating Vulkan device"))?;

        // From here on, Drop cleans up whatever has been created.
        let mut ctx = GpuContext {
            _entry: entry,
            // SAFETY: the family was requested with one queue.
            queue: unsafe { device.get_device_queue(family, 0) },
            instance: instance.take(),
            device,
            memory,
            max_storage_range: props.limits.max_storage_buffer_range as u64,
            chunks_supported,
            tile_bits: choose_tile_bits(props.limits.max_compute_shared_memory_size),
            name,
            command_pool: vk::CommandPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            lock: Mutex::new(()),
        };
        ctx.create_objects(family)?;
        Ok(Arc::new(ctx))
    }

    /// The best GPU with 64-bit floats and a compute queue: discrete first.
    fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), Error> {
        // SAFETY: plain queries on a valid instance.
        let physicals =
            unsafe { instance.enumerate_physical_devices() }.map_err(gpu_err("listing GPUs"))?;
        let mut best: Option<(vk::PhysicalDevice, u32, u32)> = None;
        for p in physicals {
            let (features, props, families) = unsafe {
                (
                    instance.get_physical_device_features(p),
                    instance.get_physical_device_properties(p),
                    instance.get_physical_device_queue_family_properties(p),
                )
            };
            let Some(family) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            // Bindings 0 and 3 hold MAX_CHUNKS state buffers each.
            let buffers = 2 * MAX_CHUNKS as u32 + 2;
            if features.shader_float64 != vk::TRUE
                || props.api_version < vk::API_VERSION_1_2
                || props.limits.max_per_stage_descriptor_storage_buffers < buffers
                || props.limits.max_descriptor_set_storage_buffers < buffers
            {
                continue;
            }
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            if best.is_none_or(|(_, _, r)| rank > r) {
                best = Some((p, family as u32, rank));
            }
        }
        best.map(|(p, f, _)| (p, f))
            .ok_or_else(|| Error::Gpu("no Vulkan 1.2 GPU with 64-bit float support".into()))
    }

    fn create_objects(&mut self, family: u32) -> Result<(), Error> {
        let d = &self.device;
        // SAFETY (whole function): valid create infos on a live device; every
        // object is stored on `self` immediately so Drop can destroy it.
        unsafe {
            self.command_pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(gpu_err("creating command pool"))?;

            // 0: state chunks (single-buffer kernels use the first);
            // 1: parameters; 2: reduction output; 3: a partner state's
            // chunks, for kernels combining two states.
            let bindings: Vec<_> = [MAX_CHUNKS as u32, 1, 1, MAX_CHUNKS as u32]
                .into_iter()
                .enumerate()
                .map(|(b, count)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(count)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            self.descriptor_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(gpu_err("creating descriptor layout"))?;

            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<Push>() as u32)];
            let layouts = [self.descriptor_layout];
            self.pipeline_layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(gpu_err("creating pipeline layout"))?;

            let tile = TILE_SHADERS
                .iter()
                .find(|(bits, _)| *bits == self.tile_bits)
                .expect("a compiled tile size")
                .1;
            // Pipelines in kernel order, [single-buffer, chunked] each; the
            // chunked variant is null when the device can't use it.
            for (variant, spv) in SHADERS
                .into_iter()
                .chain([tile])
                .flat_map(|pair| pair.into_iter().enumerate())
            {
                if variant == 1 && !self.chunks_supported {
                    self.pipelines.push(vk::Pipeline::null());
                    continue;
                }
                let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
                    .map_err(|e| Error::Gpu(format!("reading SPIR-V: {e}")))?;
                let module = d
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                    .map_err(gpu_err("creating shader module"))?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let info = [vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(self.pipeline_layout)];
                let result = d.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
                d.destroy_shader_module(module, None);
                let pipeline = result.map_err(|(_, e)| gpu_err("creating compute pipeline")(e))?;
                self.pipelines.push(pipeline[0]);
            }

            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((2 * MAX_CHUNKS as u32 + 2) * 1024)];
            self.descriptor_pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                        .max_sets(1024)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(gpu_err("creating descriptor pool"))?;
        }
        Ok(())
    }

    /// The GPU's name, as reported by the driver.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Qubits per shared-memory tile: gates on lower qubits run in tiles.
    pub(crate) fn tile_bits(&self) -> u32 {
        self.tile_bits
    }

    /// The pipeline for `kernel`, in its chunked variant if `chunked`.
    pub(crate) fn pipeline(&self, kernel: Kernel, chunked: bool) -> vk::Pipeline {
        self.pipelines[kernel as usize * 2 + chunked as usize]
    }

    /// Whether states may span several buffers.
    pub(crate) fn chunks_supported(&self) -> bool {
        self.chunks_supported
    }

    /// The most amplitudes one state buffer may hold, as a power of two:
    /// 16-byte amplitudes within the device's storage-buffer range.
    pub(crate) fn max_chunk_bits(&self) -> u32 {
        let mut bits = 0;
        while 16u64 << (bits + 1) <= self.max_storage_range && bits < 31 {
            bits += 1;
        }
        bits
    }

    pub(crate) fn push_constants(&self, cmd: vk::CommandBuffer, push: &Push) {
        // SAFETY: the layout declares exactly this range.
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push.bytes(),
            )
        }
    }

    /// Record commands with `record`, submit them and wait for completion.
    pub(crate) fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), Error> {
        let _guard = self.lock.lock().expect("GPU lock poisoned");
        let d = &self.device;
        // SAFETY: the pool is used only under the lock; the buffer and fence
        // are freed before returning.
        unsafe {
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(gpu_err("allocating command buffer"))?[0];
            let result = (|| {
                d.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(gpu_err("recording commands"))?;
                record(cmd);
                d.end_command_buffer(cmd)
                    .map_err(gpu_err("recording commands"))?;
                let fence = d
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(gpu_err("creating fence"))?;
                let cmds = [cmd];
                let submitted = d
                    .queue_submit(
                        self.queue,
                        &[vk::SubmitInfo::default().command_buffers(&cmds)],
                        fence,
                    )
                    .and_then(|()| d.wait_for_fences(&[fence], true, u64::MAX));
                d.destroy_fence(fence, None);
                submitted.map_err(gpu_err("running GPU commands"))
            })();
            d.free_command_buffers(self.command_pool, &[cmd]);
            result
        }
    }

    /// A barrier making one dispatch's writes visible to the next.
    pub(crate) fn barrier(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
    ) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )];
        // SAFETY: recording into a command buffer in the recording state.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            )
        }
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&i| {
            bits & (1 << i) != 0
                && self.memory.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }
}

/// The largest compiled tile (16 bytes per amplitude) that fits the
/// device's shared memory, up to [`PREFERRED_TILE_BITS`]. The environment
/// variable `KETRS_GPU_TILE_BITS` overrides the preference, for tuning.
fn choose_tile_bits(shared_memory: u32) -> u32 {
    let preferred = std::env::var("KETRS_GPU_TILE_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PREFERRED_TILE_BITS);
    TILE_SHADERS
        .iter()
        .map(|(bits, _)| *bits)
        .filter(|&bits| bits <= preferred && 16u64 << bits <= shared_memory as u64)
        .max()
        .unwrap_or(TILE_SHADERS[0].0)
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // SAFETY: no buffer or submission can outlive the context, since
        // they hold an Arc to it; null handles are ignored by Vulkan.
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            for &p in &self.pipelines {
                d.destroy_pipeline(p, None);
            }
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.descriptor_layout, None);
            d.destroy_command_pool(self.command_pool, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Where a buffer's memory lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Memory {
    /// GPU memory, for the state.
    Device,
    /// Host memory the CPU writes and the GPU reads (parameters, uploads).
    Upload,
    /// Host memory the GPU writes and the CPU reads: cached, because
    /// uncached reads run at a fraction of memory speed.
    Download,
}

/// A Vulkan buffer with its own memory.
pub(crate) struct Buffer {
    ctx: Arc<GpuContext>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) size: u64,
    /// The mapping of host-visible memory.
    mapped: Option<*mut u8>,
    /// Whether the mapping needs no explicit invalidation before reads.
    coherent: bool,
}

// SAFETY: the mapping is only written through `&mut self` or read through
// `&self` while no GPU work is in flight (submissions are waited on).
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// A buffer of `size` bytes in the given memory; host memory is mapped.
    pub(crate) fn new(
        ctx: &Arc<GpuContext>,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory: Memory,
    ) -> Result<Self, Error> {
        use vk::MemoryPropertyFlags as F;
        let d = &ctx.device;
        // Acceptable property sets, best first.
        let choices: &[F] = match memory {
            Memory::Device => &[F::DEVICE_LOCAL],
            Memory::Upload => &[F::HOST_VISIBLE | F::HOST_COHERENT],
            Memory::Download => &[
                F::HOST_VISIBLE | F::HOST_CACHED | F::HOST_COHERENT,
                F::HOST_VISIBLE | F::HOST_CACHED,
                F::HOST_VISIBLE | F::HOST_COHERENT,
            ],
        };
        let host = memory != Memory::Device;
        // SAFETY: valid create infos; failures release what was created.
        unsafe {
            let buffer = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size.max(16))
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(gpu_err("creating buffer"))?;
            let req = d.get_buffer_memory_requirements(buffer);
            let Some(kind) = choices
                .iter()
                .find_map(|&flags| ctx.memory_type(req.memory_type_bits, flags))
            else {
                d.destroy_buffer(buffer, None);
                return Err(Error::Gpu("no suitable GPU memory type".into()));
            };
            let device_memory = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(kind),
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(Error::Gpu(format!(
                        "allocating {} MiB of GPU memory: {e}",
                        size >> 20
                    )));
                }
            };
            let coherent = ctx.memory.memory_types[kind as usize]
                .property_flags
                .contains(F::HOST_COHERENT);
            let mut buf = Buffer {
                ctx: ctx.clone(),
                buffer,
                memory: device_memory,
                size,
                mapped: None,
                coherent,
            };
            d.bind_buffer_memory(buffer, device_memory, 0)
                .map_err(gpu_err("binding GPU memory"))?;
            if host {
                buf.mapped = Some(
                    d.map_memory(
                        device_memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(gpu_err("mapping GPU memory"))? as *mut u8,
                );
            }
            Ok(buf)
        }
    }

    /// Copy `bytes` into a mapped buffer.
    pub(crate) fn write(&mut self, bytes: &[u8]) {
        let ptr = self.mapped.expect("host-visible buffer");
        assert!(bytes.len() as u64 <= self.size);
        // SAFETY: the mapping covers `size` bytes and no GPU work is running.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    }

    /// Copy the first `out.len()` bytes of a mapped buffer.
    pub(crate) fn read(&self, out: &mut [u8]) -> Result<(), Error> {
        let ptr = self.mapped.expect("host-visible buffer");
        assert!(out.len() as u64 <= self.size);
        if !self.coherent {
            let range = [vk::MappedMemoryRange::default()
                .memory(self.memory)
                .size(vk::WHOLE_SIZE)];
            // SAFETY: the memory is mapped and the GPU has finished with it.
            unsafe { self.ctx.device.invalidate_mapped_memory_ranges(&range) }
                .map_err(gpu_err("reading GPU memory"))?;
        }
        // SAFETY: as for `write`.
        unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len()) };
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: submissions are waited on, so the GPU no longer uses it.
        unsafe {
            if self.mapped.is_some() {
                self.ctx.device.unmap_memory(self.memory);
            }
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}
