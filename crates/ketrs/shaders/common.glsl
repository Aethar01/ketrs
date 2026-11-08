// Shared declarations for ketrs's statevector compute shaders.
//
// Amplitudes are complex numbers (cplx: real, imaginary) of type `real`:
// doubles, or floats in the SINGLE = 1 variants for GPUs without 64-bit
// floats. Each shader
// processes `units` independent pieces of work (amplitude pairs, quads or
// single amplitudes) in a grid-stride loop, so any state size fits in the
// workgroup-count limit. Gate parameters (matrix entries, cosines, …) live in
// a separate buffer of doubles at `offset`.
//
// Shaders are built in two variants. With CHUNKED = 0 the state is one
// buffer. With CHUNKED = 1 it is split over up to MAX_CHUNKS buffers of
// 2^chunk_bits amplitudes each, indexed by the top bits of the amplitude
// index, for states larger than one buffer can hold. Shaders access
// amplitudes only through load_amp and store_amp.
//
// Chunked shaders reach the buffers through their device addresses, listed
// at bindings 4 and 5, not through an array of descriptors: NVIDIA's
// compiler turns a non-uniformly indexed descriptor array into a loop over
// the distinct indices, which ran 28-qubit circuits 10-20x slower on an
// RTX 4090.

#if CHUNKED
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#endif

#if SINGLE
#define real float
#define cplx vec2
#else
#define real double
#define cplx dvec2
#endif

layout(local_size_x = 256) in;

#define MAX_CHUNKS 16

// Binding 3 is a second state of the same size, the "partner", for kernels
// that combine two states (overlaps, H|ψ⟩); otherwise it repeats binding 0.
// Chunked kernels find the two states' chunks at bindings 4 and 5 instead.
#if CHUNKED
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Chunk { cplx data[]; };
layout(std430, set = 0, binding = 4) readonly buffer Chunks { uvec2 chunks[MAX_CHUNKS]; };
layout(std430, set = 0, binding = 5) readonly buffer PartnerChunks { uvec2 partner_chunks[MAX_CHUNKS]; };
#else
layout(std430, set = 0, binding = 0) buffer State { cplx amps[]; };
layout(std430, set = 0, binding = 3) buffer Partner { cplx partner_amps[]; };
#endif
// Parameters are always doubles, written by the host as f64. Without
// 64-bit floats they are read as pairs of 32-bit words and decoded with
// integer arithmetic: integers (masks, qubits) exactly, other values
// rounded towards zero to float.
#if SINGLE
layout(std430, set = 0, binding = 1) readonly buffer Params { uint params_raw[]; };

real param(uint i) {
    uint lo = params_raw[2u * i], hi = params_raw[2u * i + 1u];
    int e = int((hi >> 20u) & 0x7FFu) - 1023;
    if (e < -126) {
        return 0.0;
    }
    uint mantissa = ((hi & 0xFFFFFu) << 3u) | (lo >> 29u);
    return uintBitsToFloat((hi & 0x80000000u) | (uint(min(e, 127) + 127) << 23u) | mantissa);
}

// A non-negative integer below 2^32 stored as a double.
uint param_uint(uint i) {
    uint lo = params_raw[2u * i], hi = params_raw[2u * i + 1u];
    int e = int((hi >> 20u) & 0x7FFu) - 1023;
    if (e < 0) {
        return 0u;
    }
    // The value is 1.f × 2^e: the leading one and the top e fraction bits.
    uint e_u = uint(e);
    uint top = e_u == 0u ? 0u
        : e_u <= 20u ? (hi & 0xFFFFFu) >> (20u - e_u)
        : ((hi & 0xFFFFFu) << (e_u - 20u)) | (lo >> (52u - e_u));
    return (1u << e_u) | top;
}
#else
layout(std430, set = 0, binding = 1) readonly buffer Params { double params[]; };

real param(uint i) {
    return params[i];
}

uint param_uint(uint i) {
    return uint(params[i]);
}
#endif

// Generic slots; each shader documents its use of a, b, c.
layout(push_constant) uniform Push {
    uint a;
    uint b;
    uint c;
    uint units;
    uint offset;
    uint chunk_bits;
} pc;

#if CHUNKED
cplx load_amp(uint i) {
    return Chunk(chunks[i >> pc.chunk_bits]).data[i & ((1u << pc.chunk_bits) - 1u)];
}
void store_amp(uint i, cplx v) {
    Chunk(chunks[i >> pc.chunk_bits]).data[i & ((1u << pc.chunk_bits) - 1u)] = v;
}
cplx load_partner(uint i) {
    return Chunk(partner_chunks[i >> pc.chunk_bits]).data[i & ((1u << pc.chunk_bits) - 1u)];
}
void store_partner(uint i, cplx v) {
    Chunk(partner_chunks[i >> pc.chunk_bits]).data[i & ((1u << pc.chunk_bits) - 1u)] = v;
}
#else
cplx load_amp(uint i) {
    return amps[i];
}
void store_amp(uint i, cplx v) {
    amps[i] = v;
}
cplx load_partner(uint i) {
    return partner_amps[i];
}
void store_partner(uint i, cplx v) {
    partner_amps[i] = v;
}
#endif

cplx cmul(cplx x, cplx y) {
    return cplx(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

// Complex parameter number i.
cplx cparam(uint i) {
    return cplx(param(pc.offset + 2u * i), param(pc.offset + 2u * i + 1u));
}

// k with a zero bit inserted at position `bit`.
uint insert_zero_bit(uint k, uint bit) {
    return ((k >> bit) << (bit + 1u)) | (k & ((1u << bit) - 1u));
}

uint parity(uint x) {
    return uint(bitCount(x)) & 1u;
}

#define FOR_EACH_UNIT(k) \
    for (uint k = gl_GlobalInvocationID.x; k < pc.units; k += gl_NumWorkGroups.x * 256u)
