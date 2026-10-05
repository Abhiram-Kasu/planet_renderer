struct FragmentInput {
    float3 normal : COLOR0;
    float displacement : TEXCOORD0;
    float3 world_position : TEXCOORD1;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[[vk::binding(0, 1)]] cbuffer TerrainColorRange {
    float4 displacement_range;
    float4 low_color;
    float4 high_color;
    float4 lighting_parameters;
};

[[vk::binding(1, 1)]] StructuredBuffer<float4> terrain_samples;

[[vk::binding(2, 1)]] cbuffer TerrainParameters {
    float4 sphere_parameters;
    float4 terrain_parameters;
    float4 mesh_parameters;
};

[[vk::binding(3, 1)]] cbuffer AnimationParameters {
    uint terrain_seed;
    float elapsed_seconds;
    float animation_speed;
    float padding;
};

float terrain_height(float3 direction) {
    float weighted_height = 0.0;
    float total_weight = 0.0;
    for (uint sample_index = 0; sample_index < 100; ++sample_index) {
        float similarity = dot(direction, terrain_samples[sample_index].xyz);
        float angular_distance = max(1.0 - similarity, 0.0);
        float softened_distance = angular_distance + terrain_parameters.y;
        float weight = 1.0 / (softened_distance * softened_distance);
        weighted_height += terrain_samples[sample_index].w * weight;
        total_weight += weight;
    }
    return terrain_parameters.x * weighted_height / total_weight;
}

float fragment_displacement(float3 world_position) {
    float3 direction = normalize(world_position - sphere_parameters.xyz);
    float phase = elapsed_seconds * animation_speed * 2.0
        + dot(direction, float3(3.1, 2.3, 2.7));
    float living_motion = sin(phase) * terrain_parameters.x * 0.25;
    return abs(terrain_height(direction) + living_motion);
}

FragmentOutput fs_main(FragmentInput input) {
    FragmentOutput output;
    float range = max(displacement_range.y - displacement_range.x, 1.0e-6);
    float displacement = fragment_displacement(input.world_position);
    float gradient_position = saturate((displacement - displacement_range.x) / range);
    float3 surface_color = lerp(low_color.xyz, high_color.xyz, gradient_position);
    float3 normal = normalize(input.normal);
    float3 light_direction = normalize(lighting_parameters.xyz);
    float diffuse = max(dot(normal, light_direction), 0.0);
    float lighting = displacement_range.z + displacement_range.w * diffuse;
    output.color = float4(surface_color * lighting, 1.0);
    return output;
}
