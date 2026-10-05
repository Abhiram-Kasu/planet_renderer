struct VertexInput {
    float3 position : POSITION;
    float3 normal : COLOR0;
    float displacement : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal : COLOR0;
    float displacement : TEXCOORD0;
    float3 world_position : TEXCOORD1;
};
[[vk::binding(0, 0)]] cbuffer CameraData {
    float4x4 view_projection;
};

VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = mul(view_projection, float4(input.position, 1.0));
    output.normal = input.normal;
    output.displacement = input.displacement;
    output.world_position = input.position;
    return output;
}
