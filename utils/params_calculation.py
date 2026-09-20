# Developed by Liguang Zhou

from thop import profile


def thop_params_calculation(network, inputs):
    flops, params = profile(network, inputs=(inputs,))

    return flops, params

def params_calculation(FPAM_net, gcn_max_med_model):
    fpam_total_params = sum(param.numel() for param in FPAM_net.parameters())
    gcn_max_med_total_params = sum(param.numel() for param in gcn_max_med_model.parameters())
    print('# FPAM_net parameters:', fpam_total_params)
    print('# gcn_max_med_model parameters:', gcn_max_med_total_params)
    total_params = fpam_total_params + gcn_max_med_total_params
    print('total_params:', total_params)

    return total_params, fpam_total_params, gcn_max_med_total_params