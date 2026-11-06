// Shared declarations for ketrs's statevector compute shaders.
//
// Amplitudes are complex doubles (dvec2: real, imaginary). Each shader
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

#extension GL_EXT_nonuniform_qualifier : enable

layout(local_size_x = 256) in;

#define MAX_CHUNKS 8

// Binding 3 is a second state of the same size, the "partner", for kernels
// that combine two states (overlaps, H|ψ⟩); otherwise it repeats binding 0.
#if CHUNKED
layout(std430, set = 0, binding = 0) buffer State { dvec2 data[]; } chunks[MAX_CHUNKS];
layout(std430, set = 0, binding = 3) buffer Partner { dvec2 data[]; } partner_chunks[MAX_CHUNKS];
#else
layout(std430, set = 0, binding = 0) buffer State { dvec2 amps[]; };
layout(std430, set = 0, binding = 3) buffer Partner { dvec2 partner_amps[]; };
#endif
layout(std430, set = 0, binding = 1) readonly buffer Params { double params[]; };

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
dvec2 load_amp(uint i) {
    return chunks[nonuniformEXT(i >> pc.chunk_bits)].data[i & ((1u << pc.chunk_bits) - 1u)];
}
void store_amp(uint i, dvec2 v) {
    chunks[nonuniformEXT(i >> pc.chunk_bits)].data[i & ((1u << pc.chunk_bits) - 1u)] = v;
}
dvec2 load_partner(uint i) {
    return partner_chunks[nonuniformEXT(i >> pc.chunk_bits)].data[i & ((1u << pc.chunk_bits) - 1u)];
}
void store_partner(uint i, dvec2 v) {
    partner_chunks[nonuniformEXT(i >> pc.chunk_bits)].data[i & ((1u << pc.chunk_bits) - 1u)] = v;
}
#else
dvec2 load_amp(uint i) {
    return amps[i];
}
void store_amp(uint i, dvec2 v) {
    amps[i] = v;
}
dvec2 load_partner(uint i) {
    return partner_amps[i];
}
void store_partner(uint i, dvec2 v) {
    partner_amps[i] = v;
}
#endif

dvec2 cmul(dvec2 x, dvec2 y) {
    return dvec2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

// Complex parameter number i.
dvec2 cparam(uint i) {
    return dvec2(params[pc.offset + 2u * i], params[pc.offset + 2u * i + 1u]);
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
