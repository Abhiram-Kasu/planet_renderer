struct TerrainVertex {
    float3 position;
    float padding;
    float3 normal;
    float displacement;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> base_directions;
[[vk::binding(1, 0)]] RWStructuredBuffer<TerrainVertex> terrain_vertices;
[[vk::binding(2, 0)]] StructuredBuffer<float4> terrain_samples;

[[vk::binding(3, 0)]] cbuffer TerrainParameters {
    float4 sphere_parameters;
    float4 terrain_parameters;
    float4 mesh_parameters;
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

float3 displaced_position(float3 direction) {
    float height = terrain_height(direction);
    return sphere_parameters.xyz + direction * (sphere_parameters.w + height);
}

[numthreads(64, 1, 1)]
void cs_main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint vertex_index = dispatch_id.x;
    uint vertex_count = uint(mesh_parameters.x);
    if (vertex_index >= vertex_count) return;

    uint longitude_segments = uint(terrain_parameters.w);
    uint grid_width = longitude_segments + 1;
    uint latitude_segments = uint(terrain_parameters.z);
    uint latitude = vertex_index / grid_width;
    uint longitude = vertex_index % grid_width;
    uint wrapped_longitude = longitude % longitude_segments;

    float3 direction = normalize(base_directions[vertex_index].xyz);
    float height = terrain_height(direction);
    float3 position = sphere_parameters.xyz + direction * (sphere_parameters.w + height);

    uint before_latitude = latitude > 0 ? latitude - 1 : 0;
    uint after_latitude = min(latitude + 1, latitude_segments);
    uint before_longitude = (wrapped_longitude + longitude_segments - 1) % longitude_segments;
    uint after_longitude = (wrapped_longitude + 1) % longitude_segments;
    uint grid_row = latitude * grid_width;
    float3 before_lat = displaced_position(base_directions[before_latitude * grid_width + wrapped_longitude].xyz);
    float3 after_lat = displaced_position(base_directions[after_latitude * grid_width + wrapped_longitude].xyz);
    float3 before_lon = displaced_position(base_directions[grid_row + before_longitude].xyz);
    float3 after_lon = displaced_position(base_directions[grid_row + after_longitude].xyz);

    float3 normal = cross(after_lat - before_lat, after_lon - before_lon);
    if (dot(normal, normal) < 1.0e-12) {
        normal = direction;
    } else {
        normal = normalize(normal);
        if (dot(normal, direction) < 0.0) normal = -normal;
    }

    TerrainVertex output;
    output.position = position;
    output.padding = 0.0;
    output.normal = normal;
    output.displacement = abs(height);
    terrain_vertices[vertex_index] = output;
}
