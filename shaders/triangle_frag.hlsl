struct FragmentInput {
    float3 gradient : COLOR0;
};

float4 fs_main(FragmentInput input) : SV_Target0 {
    return float4(input.gradient, 1.0);
}
