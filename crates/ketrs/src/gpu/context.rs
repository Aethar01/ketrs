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
    /// A Lanczos step's α or β from overlap partial sums, written into the
    /// parameters (`lanczos.comp`).
    Lanczos = 10,
}

impl Kernel {
    /// Workgroups to dispatch for `units` units of work. Tiles and blocks
    /// take one workgroup each; reductions are capped so their partial sums
    /// stay small; other kernels use one thread per unit.
    pub(crate) fn groups(self, units: u32) -> u32 {
        let groups = match self {
            Kernel::Tile | Kernel::Probs => units,
            Kernel::Lanczos => 1,
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

/// Each kernel before `Tile`, as `[double, single][single-buffer, chunked]`
/// variants.
const SHADERS: [[[&[u8]; 2]; 2]; 9] = [
    [
        [spirv!("pair_0_0"), spirv!("pair_1_0")],
        [spirv!("pair_0_1"), spirv!("pair_1_1")],
    ],
    [
        [spirv!("quad_0_0"), spirv!("quad_1_0")],
        [spirv!("quad_0_1"), spirv!("quad_1_1")],
    ],
    [
        [spirv!("pauli_0_0"), spirv!("pauli_1_0")],
        [spirv!("pauli_0_1"), spirv!("pauli_1_1")],
    ],
    [
        [spirv!("phase_0_0"), spirv!("phase_1_0")],
        [spirv!("phase_0_1"), spirv!("phase_1_1")],
    ],
    [
        [spirv!("expect_0_0"), spirv!("expect_1_0")],
        [spirv!("expect_0_1"), spirv!("expect_1_1")],
    ],
    [
        [spirv!("probs_0_0"), spirv!("probs_1_0")],
        [spirv!("probs_0_1"), spirv!("probs_1_1")],
    ],
    [
        [spirv!("overlap_0_0"), spirv!("overlap_1_0")],
        [spirv!("overlap_0_1"), spirv!("overlap_1_1")],
    ],
    [
        [spirv!("hamil_0_0"), spirv!("hamil_1_0")],
        [spirv!("hamil_0_1"), spirv!("hamil_1_1")],
    ],
    [
        [spirv!("axpy_0_0"), spirv!("axpy_1_0")],
        [spirv!("axpy_0_1"), spirv!("axpy_1_1")],
    ],
];

/// The Lanczos kernel, after the tile kernel, as `[double, single]
/// [single-buffer, chunked]` variants.
const LANCZOS_SHADERS: [[&[u8]; 2]; 2] = [
    [spirv!("lanczos_0_0"), spirv!("lanczos_1_0")],
    [spirv!("lanczos_0_1"), spirv!("lanczos_1_1")],
];

/// The tile kernel for 2^bits-amplitude tiles, by bits, as
/// `[double, single][single-buffer, chunked]` variants.
#[allow(clippy::type_complexity)]
const TILE_SHADERS: [(u32, [[&[u8]; 2]; 2]); 3] = [
    (
        10,
        [
            [spirv!("tile_0_0_10"), spirv!("tile_1_0_10")],
            [spirv!("tile_0_1_10"), spirv!("tile_1_1_10")],
        ],
    ),
    (
        11,
        [
            [spirv!("tile_0_0_11"), spirv!("tile_1_0_11")],
            [spirv!("tile_0_1_11"), spirv!("tile_1_1_11")],
        ],
    ),
    (
        12,
        [
            [spirv!("tile_0_0_12"), spirv!("tile_1_0_12")],
            [spirv!("tile_0_1_12"), spirv!("tile_1_1_12")],
        ],
    ),
];

/// The MPS contraction kernels (`mps_*.comp`), which take every buffer by
/// device address in their push constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MpsKernel {
    /// Batched, strided complex matrix product.
    Gemm = 0,
    /// MPO channels and physical indices mixed by a sparse table.
    Mix = 1,
    /// Sum of the partial products of a k-split `Gemm`.
    Sum = 2,
    /// Dot products and vector updates of a Lanczos recurrence.
    Krylov = 3,
    /// Column steps of a blocked Householder QR.
    Qr = 4,
    /// Grouped products of blocks, in 8×8 output tiles.
    BlockGemm8 = 5,
    /// Grouped products of blocks, in 16×16 output tiles.
    BlockGemm16 = 6,
    /// Sums of scaled blocks into disjoint segments.
    BlockMix = 7,
}

const MPS_SHADERS: [&[u8]; 8] = [
    spirv!("mps_gemm"),
    spirv!("mps_mix"),
    spirv!("mps_sum"),
    spirv!("mps_krylov"),
    spirv!("mps_qr"),
    spirv!("mps_bgemm_8"),
    spirv!("mps_bgemm_16"),
    spirv!("mps_bmix"),
];

/// Push-constant bytes of the MPS kernels' layout (Vulkan's guaranteed
/// minimum).
pub(crate) const MPS_PUSH_BYTES: u32 = 128;

/// State buffers a chunked kernel can address (`MAX_CHUNKS` in
/// `shaders/common.glsl`).
pub(crate) const MAX_CHUNKS: usize = 16;

/// Storage-buffer bindings in the descriptor set (see `create_objects`).
const BINDINGS: u32 = 6;

/// Workgroups in an expectation-value dispatch, so each one writes at most
/// this many partial sums.
pub(crate) const REDUCTION_GROUPS: u32 = 1024;

/// Largest tile to use when the device allows it. Smaller tiles mean more
/// workgroups resident at once; on an RX 9060 XT, 10-qubit tiles beat 11
/// and 12 by 10–50%, and 10 fits Vulkan's guaranteed 16 KiB of shared
/// memory, so every device can use it.
const PREFERRED_TILE_BITS: u32 = 10;

/// The preference on NVIDIA GPUs, where larger tiles (fewer passes) win:
/// on an RTX 4090, 11-qubit tiles ran 28-qubit circuits 1.5–3× faster and
/// 26-qubit ones up to 1.35× faster. 12 doesn't fit its 48 KiB.
const NVIDIA_TILE_BITS: u32 = 11;

/// PCI vendor ID of NVIDIA.
const NVIDIA: u32 = 0x10de;

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
/// Creating one loads the system Vulkan driver and picks a GPU, preferring
/// a discrete card and, among equals, one with 64-bit floats. Without them,
/// states are single precision (see
/// [`GpuPrecision`](super::GpuPrecision)). Share it through the returned
/// `Arc` across any number of [`GpuStatevector`](super::GpuStatevector)s.
pub struct GpuContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    physical: vk::PhysicalDevice,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// Whether the driver reports memory budgets (`VK_EXT_memory_budget`).
    memory_budget: bool,
    max_storage_range: u64,
    /// Whether kernels can index several state buffers, for states larger
    /// than one buffer (Vulkan 1.2 buffer device addresses).
    chunks_supported: bool,
    /// Whether shaders may use 64-bit floats; otherwise only
    /// single-precision states are available.
    f64_supported: bool,
    /// Qubits per shared-memory tile.
    tile_bits: u32,
    name: String,
    command_pool: vk::CommandPool,
    pub(crate) descriptor_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    /// Layout of the MPS kernels: push constants only.
    mps_layout: vk::PipelineLayout,
    /// The MPS kernels in [`MpsKernel`] order; empty without 64-bit floats
    /// or buffer device addresses.
    mps_pipelines: Vec<vk::Pipeline>,
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
    /// Load Vulkan and open the best compute device: discrete GPUs first,
    /// and among those, ones with 64-bit floats.
    pub fn new() -> Result<Arc<Self>, Error> {
        let (entry, instance, physical, family) = Self::open_device()?;
        Self::from_opened(entry, instance, physical, family)
    }

    fn from_opened(
        entry: ash::Entry,
        instance: InstanceGuard,
        physical: vk::PhysicalDevice,
        family: u32,
    ) -> Result<Arc<Self>, Error> {
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
        // SAFETY: plain query on a valid physical device.
        let f64_supported = unsafe { instance.get().get_physical_device_features(physical) }
            .shader_float64
            == vk::TRUE;
        let features = vk::PhysicalDeviceFeatures::default().shader_float64(f64_supported);
        let chunks_supported = {
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut all = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
            // SAFETY: plain query on a Vulkan 1.2 device.
            unsafe {
                instance
                    .get()
                    .get_physical_device_features2(physical, &mut all)
            };
            v12.buffer_device_address == vk::TRUE
        };
        // SAFETY: plain query on a valid physical device.
        let device_exts = unsafe {
            instance
                .get()
                .enumerate_device_extension_properties(physical)
        }
        .unwrap_or_default();
        let has_ext = |name: &CStr| {
            device_exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let memory_budget = has_ext(c"VK_EXT_memory_budget");
        let mut extensions = Vec::new();
        if memory_budget {
            extensions.push(c"VK_EXT_memory_budget".as_ptr());
        }
        // Per the Vulkan spec, devices advertising VK_KHR_portability_subset
        // (such as MoltenVK on macOS) require enabling it at device creation.
        if has_ext(c"VK_KHR_portability_subset") {
            extensions.push(c"VK_KHR_portability_subset".as_ptr());
        }
        let mut v12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(chunks_supported);
        // SAFETY: the chosen device supports these features, extensions and
        // the family.
        let device = unsafe {
            instance.get().create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_features(&features)
                    .enabled_extension_names(&extensions)
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
            physical,
            memory,
            memory_budget,
            max_storage_range: props.limits.max_storage_buffer_range as u64,
            chunks_supported,
            f64_supported,
            tile_bits: choose_tile_bits(
                props.limits.max_compute_shared_memory_size,
                props.vendor_id,
                f64_supported,
            ),
            name,
            command_pool: vk::CommandPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            mps_layout: vk::PipelineLayout::null(),
            mps_pipelines: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            lock: Mutex::new(()),
        };
        ctx.create_objects(family)?;
        Ok(Arc::new(ctx))
    }

    /// Create a Vulkan 1.2 instance, enabling portability enumeration when
    /// the loader or driver supports it (needed for MoltenVK on macOS).
    fn create_instance(entry: &ash::Entry) -> Result<InstanceGuard, Error> {
        let app = vk::ApplicationInfo::default()
            .application_name(c"ketrs")
            .api_version(vk::API_VERSION_1_2);
        // SAFETY: plain query on a loaded Vulkan entry.
        let portability = unsafe { entry.enumerate_instance_extension_properties(None) }
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(c"VK_KHR_portability_enumeration"));
        let portability_ext = [c"VK_KHR_portability_enumeration".as_ptr()];
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        if portability {
            info = info
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
                .enabled_extension_names(&portability_ext);
        }
        // SAFETY: valid create info; the guard destroys the instance on error.
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(gpu_err("creating Vulkan instance"))?;
        Ok(InstanceGuard(Some(instance)))
    }

    /// Load Vulkan, create an instance, and pick a compute device.
    ///
    /// On macOS, tries the default Vulkan loader first, then `$VULKAN_SDK`,
    /// a bundled `libMoltenVK.dylib` next to this binary/extension, and
    /// standard Homebrew / system loader and MoltenVK paths; if a loader
    /// finds no device (e.g. installed without an ICD), falls through to
    /// the next candidate.
    #[cfg(target_os = "macos")]
    fn open_device() -> Result<(ash::Entry, InstanceGuard, vk::PhysicalDevice, u32), Error> {
        let mut last_err = None;
        for candidate in macos_vulkan_candidates() {
            // SAFETY: loads a system or bundled Vulkan / MoltenVK dylib;
            // failure is caught and tried against the next candidate.
            let entry = match &candidate {
                None => unsafe { ash::Entry::load() },
                Some(path) => unsafe { ash::Entry::load_from(path) },
            };
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    if last_err.is_none() {
                        last_err = Some(Error::Gpu(format!("no Vulkan driver found: {e}")));
                    }
                    continue;
                }
            };
            match Self::create_instance(&entry)
                .and_then(|inst| Self::pick_device(inst.get()).map(|(p, f)| (inst, p, f)))
            {
                Ok((inst, physical, family)) => return Ok((entry, inst, physical, family)),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Gpu("no Vulkan driver found".into())))
    }

    #[cfg(not(target_os = "macos"))]
    fn open_device() -> Result<(ash::Entry, InstanceGuard, vk::PhysicalDevice, u32), Error> {
        // SAFETY: loads the system Vulkan library; failure is reported.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| Error::Gpu(format!("no Vulkan driver found: {e}")))?;
        let instance = Self::create_instance(&entry)?;
        let (physical, family) = Self::pick_device(instance.get())?;
        Ok((entry, instance, physical, family))
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
            if props.api_version < vk::API_VERSION_1_2
                || props.limits.max_per_stage_descriptor_storage_buffers < BINDINGS
                || props.limits.max_descriptor_set_storage_buffers < BINDINGS
            {
                continue;
            }
            let rank = 2 * match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            } + (features.shader_float64 == vk::TRUE) as u32;
            if best.is_none_or(|(_, _, r)| rank > r) {
                best = Some((p, family as u32, rank));
            }
        }
        best.map(|(p, f, _)| (p, f))
            .ok_or_else(|| Error::Gpu("no Vulkan 1.2 compute device".into()))
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

            // 0: the state (its first chunk); 1: parameters; 2: reduction
            // output; 3: a partner state, for kernels combining two states;
            // 4 and 5: the device addresses of the state's and partner's
            // chunks, for chunked kernels.
            let bindings: Vec<_> = (0..BINDINGS)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
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
            // Pipelines in kernel order, each as [double, single] ×
            // [single-buffer, chunked]; variants the device can't use are
            // null.
            for (index, spv) in SHADERS
                .into_iter()
                .chain([tile, LANCZOS_SHADERS])
                .flat_map(|variants| variants.into_iter().flatten())
                .enumerate()
            {
                let (single, chunked) = (index / 2 % 2 == 1, index % 2 == 1);
                if (chunked && !self.chunks_supported) || (!single && !self.f64_supported) {
                    self.pipelines.push(vk::Pipeline::null());
                    continue;
                }
                let pipeline = self.compute_pipeline(spv, self.pipeline_layout)?;
                self.pipelines.push(pipeline);
            }

            if self.f64_supported && self.chunks_supported {
                let push = [vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(MPS_PUSH_BYTES)];
                self.mps_layout = d
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                        None,
                    )
                    .map_err(gpu_err("creating pipeline layout"))?;
                for spv in MPS_SHADERS {
                    let pipeline = self.compute_pipeline(spv, self.mps_layout)?;
                    self.mps_pipelines.push(pipeline);
                }
            }

            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(BINDINGS * 1024)];
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

    /// A compute pipeline from SPIR-V with entry point `main`.
    fn compute_pipeline(
        &self,
        spv: &[u8],
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, Error> {
        let d = &self.device;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
            .map_err(|e| Error::Gpu(format!("reading SPIR-V: {e}")))?;
        // SAFETY: valid create infos on a live device; the module is
        // destroyed once the pipeline holds it.
        unsafe {
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(gpu_err("creating shader module"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let result = d.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            d.destroy_shader_module(module, None);
            Ok(result.map_err(|(_, e)| gpu_err("creating compute pipeline")(e))?[0])
        }
    }

    /// Whether the MPS kernels are available: they need 64-bit floats and
    /// buffer device addresses.
    pub fn supports_mps(&self) -> bool {
        !self.mps_pipelines.is_empty()
    }

    /// Record one dispatch of an MPS kernel with the given push constants
    /// (at most [`MPS_PUSH_BYTES`]).
    pub(crate) fn mps_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        kernel: MpsKernel,
        push: &[u8],
        groups: [u32; 3],
    ) {
        debug_assert!(push.len() as u32 <= MPS_PUSH_BYTES);
        // SAFETY: recording into a command buffer in the recording state,
        // with a pipeline and layout of this context.
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.mps_pipelines[kernel as usize],
            );
            self.device.cmd_push_constants(
                cmd,
                self.mps_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            self.device
                .cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        }
    }

    /// The GPU's name, as reported by the driver.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Qubits per shared-memory tile: gates on lower qubits run in tiles.
    pub(crate) fn tile_bits(&self) -> u32 {
        self.tile_bits
    }

    /// The pipeline for `kernel`, in its chunked variant if `chunked` and
    /// its single-precision one if `single`.
    /// An error if `bytes` more of memory type `kind` would exceed this
    /// process's budget, as the driver reports it, while other processes
    /// hold a noticeable share (over 1/16) of the heap. Overcommitting the
    /// GPU alone works: the driver moves this process's idle buffers to
    /// system memory. With other processes on the same GPU, an allocation
    /// past the budget can succeed only for later command submissions to
    /// fail and the device to be lost in every process (as with radv), which
    /// none can recover from.
    fn check_budget(&self, kind: u32, bytes: u64) -> Result<(), Error> {
        if !self.memory_budget {
            return Ok(());
        }
        let heap = self.memory.memory_types[kind as usize].heap_index as usize;
        let size = self.memory.memory_heaps[heap].size;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: plain query on a valid physical device supporting the
        // extension.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut props)
        };
        let (limit, used) = (budget.heap_budget[heap], budget.heap_usage[heap]);
        if past_shared_budget(size, limit, used, bytes) {
            return Err(Error::Gpu(format!(
                "not enough GPU memory: {} MiB more needed, {} MiB left of this process's \
                 {} MiB budget (other processes use {} MiB of the GPU's {} MiB)",
                bytes >> 20,
                limit.saturating_sub(used) >> 20,
                limit >> 20,
                size.saturating_sub(limit) >> 20,
                size >> 20
            )));
        }
        Ok(())
    }

    pub(crate) fn pipeline(&self, kernel: Kernel, chunked: bool, single: bool) -> vk::Pipeline {
        self.pipelines[kernel as usize * 4 + single as usize * 2 + chunked as usize]
    }

    /// Whether the device has 64-bit floats, so double-precision states.
    pub fn supports_f64(&self) -> bool {
        self.f64_supported
    }

    /// Whether states may span several buffers.
    pub(crate) fn chunks_supported(&self) -> bool {
        self.chunks_supported
    }

    /// The most amplitudes of `bytes` each one state buffer may hold, as a
    /// power of two, within the device's storage-buffer range.
    pub(crate) fn max_chunk_bits(&self, bytes: u64) -> u32 {
        let mut bits = 0;
        while bytes << (bits + 1) <= self.max_storage_range && bits < 31 {
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

/// Candidate Vulkan / MoltenVK library paths on macOS, in preference order.
/// `None` tries `ash::Entry::load()` (default dyld search).
#[cfg(target_os = "macos")]
fn macos_vulkan_candidates() -> Vec<Option<std::path::PathBuf>> {
    use std::path::PathBuf;

    let mut out = vec![None];
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let lib = PathBuf::from(sdk).join("lib");
        out.push(Some(lib.join("libvulkan.1.dylib")));
        out.push(Some(lib.join("libMoltenVK.dylib")));
    }
    if let Some(dir) = current_module_dir() {
        out.push(Some(dir.join("libMoltenVK.dylib")));
    }
    for path in [
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
        "/Library/Frameworks/MoltenVK.framework/MoltenVK",
    ] {
        out.push(Some(PathBuf::from(path)));
    }
    out
}

/// Directory containing the shared library or executable holding this code,
/// so macOS wheels can find a bundled `libMoltenVK.dylib` next to `_ketrs`.
#[cfg(target_os = "macos")]
fn current_module_dir() -> Option<std::path::PathBuf> {
    use std::ffi::{c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    unsafe extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }

    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    // SAFETY: `current_module_dir` is a valid code pointer in this image and
    // `info` is a valid out-parameter for `dladdr`.
    let ok = unsafe { dladdr(current_module_dir as *const c_void, &mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: `dli_fname` is a NUL-terminated path string returned by dyld.
    let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
    std::path::Path::new(std::ffi::OsStr::from_bytes(bytes))
        .parent()
        .map(std::path::Path::to_path_buf)
}

/// The largest compiled tile that fits the device's shared memory, up to
/// [`PREFERRED_TILE_BITS`] (or [`NVIDIA_TILE_BITS`]). The environment
/// variable `KETRS_GPU_TILE_BITS` overrides the preference, for tuning.
fn choose_tile_bits(shared_memory: u32, vendor: u32, f64_supported: bool) -> u32 {
    let preferred = std::env::var("KETRS_GPU_TILE_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if vendor == NVIDIA {
            NVIDIA_TILE_BITS
        } else {
            PREFERRED_TILE_BITS
        });
    let amp_bytes = if f64_supported { 16u64 } else { 8u64 };
    TILE_SHADERS
        .iter()
        .map(|(bits, _)| *bits)
        .filter(|&bits| bits <= preferred && amp_bytes << bits <= shared_memory as u64)
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
            for &p in self.pipelines.iter().chain(&self.mps_pipelines) {
                d.destroy_pipeline(p, None);
            }
            d.destroy_pipeline_layout(self.mps_layout, None);
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

/// Whether `bytes` more on a heap of `size` bytes would pass this process's
/// `limit` (having `used` of it) while other processes hold over 1/16 of
/// the heap. A zero limit means the driver doesn't know.
fn past_shared_budget(size: u64, limit: u64, used: u64, bytes: u64) -> bool {
    let shared = limit > 0 && limit < size - size / 16;
    shared && used.saturating_add(bytes) > limit
}

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
            if memory == Memory::Device
                && let Err(e) = ctx.check_budget(kind, req.size)
            {
                d.destroy_buffer(buffer, None);
                return Err(e);
            }
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(kind);
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                info = info.push_next(&mut flags);
            }
            let device_memory = match d.allocate_memory(&info, None) {
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

    /// The first `len` bytes of a mapped buffer, once the GPU has finished
    /// writing it.
    pub(crate) fn mapped(&self, len: usize) -> Result<&[u8], Error> {
        let ptr = self.mapped.expect("host-visible buffer");
        assert!(len as u64 <= self.size);
        if !self.coherent {
            let range = [vk::MappedMemoryRange::default()
                .memory(self.memory)
                .size(vk::WHOLE_SIZE)];
            // SAFETY: the memory is mapped and the GPU has finished with it.
            unsafe { self.ctx.device.invalidate_mapped_memory_ranges(&range) }
                .map_err(gpu_err("reading GPU memory"))?;
        }
        // SAFETY: the mapping covers `size` bytes, no GPU work is running,
        // and writes need `&mut self`.
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// The buffer's GPU address, for a buffer created with
    /// `SHADER_DEVICE_ADDRESS` usage.
    pub(crate) fn device_address(&self) -> u64 {
        // SAFETY: the buffer is live and was created with that usage.
        unsafe {
            self.ctx.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
            )
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_binds_only_when_other_processes_share_the_gpu() {
        let gib = 1u64 << 30;
        // Alone (the budget is about the whole heap): overcommitting is left
        // to the driver, which moves idle buffers to system memory.
        assert!(!past_shared_budget(16 * gib, 16 * gib, 15 * gib, 4 * gib));
        // Another process holds 8 GiB: past the budget is an error.
        assert!(past_shared_budget(16 * gib, 8 * gib, 7 * gib, 2 * gib));
        assert!(!past_shared_budget(16 * gib, 8 * gib, 5 * gib, 2 * gib));
        // No budget reported.
        assert!(!past_shared_budget(16 * gib, 0, 15 * gib, 4 * gib));
    }

    #[test]
    fn impossible_allocations_fail_cleanly() {
        let Ok(ctx) = GpuContext::new() else { return };
        let buffer = Buffer::new(
            &ctx,
            1 << 60,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Memory::Device,
        );
        assert!(buffer.is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn direct_moltenvk_loads_without_vulkan_loader() {
        let Some(path) = [
            "/opt/homebrew/lib/libMoltenVK.dylib",
            "/usr/local/lib/libMoltenVK.dylib",
        ]
        .into_iter()
        .find(|p| std::path::Path::new(p).exists()) else {
            return;
        };
        assert!(current_module_dir().is_some());
        // Load MoltenVK directly (without libvulkan.dylib), as a bundled
        // macOS Python wheel does on machines with no Vulkan SDK installed.
        let entry = unsafe { ash::Entry::load_from(path) }.expect("load libMoltenVK.dylib");
        let instance = GpuContext::create_instance(&entry).expect("create Vulkan instance");
        let (physical, family) =
            GpuContext::pick_device(instance.get()).expect("pick MoltenVK device");
        let ctx = GpuContext::from_opened(entry, instance, physical, family)
            .expect("create GpuContext directly from libMoltenVK.dylib");
        assert!(!ctx.device_name().is_empty());
        let mut bell = crate::Circuit::new(2);
        bell.h(0).cx(0, 1);
        let state = crate::GpuStatevector::from_circuit(&ctx, &bell).expect("run Bell circuit");
        let amps = state.amplitudes().expect("download amplitudes");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amps[0].re - s).abs() < 1e-6);
        assert!(amps[1].norm() < 1e-6);
        assert!(amps[2].norm() < 1e-6);
        assert!((amps[3].re - s).abs() < 1e-6);
    }
}
