[[vk::binding(0, 0)]] RWStructuredBuffer<float4> terrain_samples;

[[vk::binding(1, 0)]] cbuffer AnimationParameters {
    uint terrain_seed;
    float elapsed_seconds;
    float animation_speed;
    float padding;
};

uint hash_value(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return value;
}

float random_value(uint index, uint lane) {
    uint value = hash_value(terrain_seed ^ (index * 0x9e3779b9) ^ (lane * 0x85ebca6b));
    return float(value & 0x00ffffff) / 16777216.0;
}

float3 rotate_around_axis(float3 v, float3 axis, float angle) {
    float cosine = cos(angle);
    float sine = sin(angle);
    return v * cosine
        + cross(axis, v) * sine
        + axis * dot(axis, v) * (1.0 - cosine);
}

[numthreads(64, 1, 1)]
void cs_main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint sample_index = dispatch_id.x;
    if (sample_index >= 100) return;

    float z = random_value(sample_index, 0) * 2.0 - 1.0;
    float longitude = random_value(sample_index, 1) * 6.28318530718;
    float radial = sqrt(max(1.0 - z * z, 0.0));
    float3 direction = float3(radial * cos(longitude), z, radial * sin(longitude));

    float axis_z = random_value(sample_index, 2) * 2.0 - 1.0;
    float axis_angle = random_value(sample_index, 3) * 6.28318530718;
    float axis_radial = sqrt(max(1.0 - axis_z * axis_z, 0.0));
    float3 axis = normalize(float3(axis_radial * cos(axis_angle), axis_z, axis_radial * sin(axis_angle)));
    float phase = random_value(sample_index, 6) * 6.28318530718;
    float3 global_flow = rotate_around_axis(
        direction,
        float3(0.0, 1.0, 0.0),
        elapsed_seconds * animation_speed * 0.08
    );
    float local_sway = 0.025 * sin(elapsed_seconds * animation_speed * 0.7 + phase);
    direction = normalize(rotate_around_axis(global_flow, axis, local_sway));

    float base_height = random_value(sample_index, 5) * 2.0 - 1.0;
    float height_motion = 0.9 + 0.1 * sin(elapsed_seconds * animation_speed + phase);
    terrain_samples[sample_index] = float4(direction, base_height * height_motion);
}
