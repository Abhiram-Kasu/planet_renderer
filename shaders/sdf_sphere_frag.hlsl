struct FragmentInput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[[vk::binding(0, 0)]] cbuffer CameraData {
    float4x4 inverse_view_projection;
};

[[vk::binding(1, 0)]] cbuffer SphereSettings {
    float4 sphere_parameters;
    float4 march_parameters;
    float4 color_parameters;
    float4 lighting_parameters;
    float4 pattern_color;
    float4 pattern_parameters;
};

float sphere_sdf(float3 p) {
    return length(p - sphere_parameters.xyz) - sphere_parameters.w;
}

FragmentOutput fs_main(FragmentInput input) {
    FragmentOutput output;
    float2 ndc = input.uv * 2.0 - 1.0;
    float4 near_h = mul(inverse_view_projection, float4(ndc, 0.0, 1.0));
    float4 far_h = mul(inverse_view_projection, float4(ndc, 1.0, 1.0));
    float3 ray_origin = near_h.xyz / near_h.w;
    float3 ray_end = far_h.xyz / far_h.w;
    float3 ray_direction = normalize(ray_end - ray_origin);

    float distance_along_ray = 0.0;
    bool hit = false;
    float3 hit_point;
    for (int step = 0; step < int(march_parameters.x); ++step) {
        hit_point = ray_origin + ray_direction * distance_along_ray;
        float distance_to_surface = sphere_sdf(hit_point);
        if (distance_to_surface < march_parameters.y) {
            hit = true;
            break;
        }
        distance_along_ray += distance_to_surface;
        if (distance_along_ray > march_parameters.z) break;
    }

    if (!hit) {
        discard;
    }

    float cosine = cos(march_parameters.w);
    float sine = sin(march_parameters.w);
    float3 local_position = (hit_point - sphere_parameters.xyz) / sphere_parameters.w;
    float3 rotated_position = float3(
        cosine * local_position.x - sine * local_position.z,
        local_position.y,
        sine * local_position.x + cosine * local_position.z
    );
    float longitude = atan2(rotated_position.z, rotated_position.x);
    float latitude = asin(clamp(rotated_position.y, -1.0, 1.0));
    float pattern_value = sin(
        longitude * pattern_parameters.x + sin(latitude * pattern_parameters.y)
    );
    float pattern_mask = smoothstep(
        pattern_parameters.z,
        pattern_parameters.z + pattern_parameters.w,
        pattern_value
    );
    float3 normal = normalize(hit_point - sphere_parameters.xyz);
    float3 light_direction = normalize(lighting_parameters.xyz);
    float diffuse = max(dot(normal, light_direction), 0.0);
    float lighting = color_parameters.w + diffuse * lighting_parameters.w;
    float3 surface_color = lerp(color_parameters.xyz, pattern_color.xyz,
                                pattern_mask * pattern_color.w);
    output.color = float4(surface_color * lighting, 1.0);
    return output;
}
