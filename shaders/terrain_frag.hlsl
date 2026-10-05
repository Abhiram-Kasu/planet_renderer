struct FragmentInput {
    float3 normal : COLOR0;
    float displacement : TEXCOORD0;
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

FragmentOutput fs_main(FragmentInput input) {
    FragmentOutput output;
    float range = max(displacement_range.y - displacement_range.x, 1.0e-6);
    float gradient_position = saturate((input.displacement - displacement_range.x) / range);
    float3 surface_color = lerp(low_color.xyz, high_color.xyz, gradient_position);
    float3 normal = normalize(input.normal);
    float3 light_direction = normalize(lighting_parameters.xyz);
    float diffuse = max(dot(normal, light_direction), 0.0);
    float lighting = displacement_range.z + displacement_range.w * diffuse;
    output.color = float4(surface_color * lighting, 1.0);
    return output;
}
