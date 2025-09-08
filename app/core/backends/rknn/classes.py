import ctypes
import os
import config

# Load RKNN runtime
rknn_library_path = os.path.join(config.get_path("lib"), "librknnrt.so")
rknn_lib = ctypes.CDLL(rknn_library_path)

# Constants from rknn_api.h
RKNN_SUCC = 0
RKNN_QUERY_IN_OUT_NUM = 0  # query the number of input & output tensor.
RKNN_QUERY_INPUT_ATTR = 1  # query the attribute of input tensor.
RKNN_QUERY_OUTPUT_ATTR = 2  # query the attribute of output tensor.
RKNN_QUERY_PERF_DETAIL = 3  # query the detail performance need set RKNN_FLAG_COLLECT_PERF_MASK when call rknn_init this query needs to be valid after rknn_outputs_get.
RKNN_QUERY_PERF_RUN = 4  # query the time of run this query needs to be valid after rknn_outputs_get.
RKNN_QUERY_SDK_VERSION = 5  # query the sdk & driver version
RKNN_QUERY_MEM_SIZE = 6  # query the weight & internal memory size
RKNN_QUERY_CUSTOM_STRING = 7  # query the custom string
RKNN_QUERY_NATIVE_INPUT_ATTR = 8  # query the attribute of native input tensor.
RKNN_QUERY_NATIVE_OUTPUT_ATTR = 9  # query the attribute of native output tensor.
RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR = 8  # query the attribute of native input tensor.
RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR = 9  # query the attribute of native output tensor.
RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR = 10  # query the attribute of native input tensor.
RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR = 11  # query the attribute of native output tensor.
RKNN_QUERY_DEVICE_MEM_INFO = 12  # query the attribute of rknn memory information.
RKNN_QUERY_INPUT_DYNAMIC_RANGE = 13  # query the dynamic shape range of rknn input tensor.
RKNN_QUERY_CURRENT_INPUT_ATTR = 14  # query the current shape of rknn input tensor only valid for dynamic rknn model
RKNN_QUERY_CURRENT_OUTPUT_ATTR = 15  # query the current shape of rknn output tensor only valid for dynamic rknn model
RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR = 16  # query the current native shape of rknn input tensor only valid for dynamic rknn model
RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR = 17  # query the current native shape of rknn output tensor only valid for dynamic rknn model

RKNN_NPU_CORE_AUTO = 0  # default, run on NPU core randomly.
RKNN_NPU_CORE_0 = 1  # run on NPU core 0.
RKNN_NPU_CORE_1 = 2  # run on NPU core 1.
RKNN_NPU_CORE_2 = 4  # run on NPU core 2.
RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 + RKNN_NPU_CORE_1  # run on NPU core 0 and core 1.
RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 + RKNN_NPU_CORE_2  # run on NPU core 0 and core 1 and core 2.
RKNN_NPU_CORE_ALL = 0xffff  # auto choice, run on NPU cores depending on platform

# typedef RKNNContext (non-arm: uint64_t)
RKNNContext = ctypes.c_uint64


class RKNNInitExtend(ctypes.Structure):
    _fields_ = [
        ("ctx", ctypes.c_uint64),
        ("real_model_offset", ctypes.c_int32),
        ("real_model_size", ctypes.c_uint32),
        ("model_buffer_fd", ctypes.c_int32),
        ("model_buffer_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 112)]


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


# small helper context object
class RknnAppContext:
    def __init__(self, base_domain_id=0):
        self.rknn_ctx = RKNNContext(base_domain_id)
        self.io_num = None
        self.input_attrs = []
        self.output_attrs = []
        self.model_height = 0
        self.model_width = 0
        self.model_channel = 0