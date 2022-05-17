ENABLE_ALL_FEATURES = False


if ENABLE_ALL_FEATURES:
    FEATURE_NAMES = [
        'latency', 'cpu_use', 'mem_use_percent', 'mem_use_amount',
        'file_write_rate', 'file_read_rate',
        'net_send_rate', 'net_receive_rate', 'http_status'
    ]
else:
    FEATURE_NAMES = [
        'latency', 'http_status'
    ]

FAULT_TYPES = {'delay', 'abort', 'cpu'}

INVOLVED_SERVICES = [
    'route-plan',
    'food',
    'config',
    'order',
    'seat',
    'train',
    'travel-plan',
    'user',
    'route',
    'ticketinfo',
    'verification-code',
    'price',
    'contacts',
    'cancel',
    'travel2',
    'assurance',
    'preserve',
    'basic',
    'auth',
    'security',
    'consign',
    'food-map',
    'travel',
    'station',
    'ui-dashboard',
    'preserve-other',
    'order-other',
    'inside-payment',
    'execute',
    'payment',
    'admin-order',
    'admin-basic-info',
    'gateway',
    'admin-route',
    'admin-travel',
    'notification',
    'admin-user',
    'istio-mixer',
    'kibana',
    'jaeger-query',
]


SERVICE2IDX = {service: idx for idx, service in enumerate(INVOLVED_SERVICES)}

EXP_NOISE_LIST = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
# EXP_NOISE_LIST = [0]
