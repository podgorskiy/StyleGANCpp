# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
sys.path.append('stylegan')
sys.path.append('stylegan/dnnlib')
sys.path.append('tensor4/tensor4')

import struct
import dnnlib
import dnnlib.tflib
import pickle
import numpy as np


def write_blob(model, filename=None):
    if filename is None:
        filename = model.class_name + '.t4'

    with open(filename, "wb") as f:
        for w_name in model.model_dict.keys():
            w = model.model_dict[w_name]
            f.write(w_name.encode())
            f.write(bytes([0]))
            if w.dtype == np.float64:
                f.write(bytes(b'doubl'))
            if w.dtype == np.float32:
                f.write(bytes(b'float'))
            if w.dtype == np.int32:
                f.write(bytes(b'int32'))
            if w.dtype == np.int16:
                f.write(bytes(b'int16'))
            f.write(struct.pack("b", w.ndim))
            for i in range(w.ndim):
                f.write(struct.pack("i", w.shape[i]))
            b = w.tobytes()
            f.write(struct.pack("Q", len(b)))
            f.write(struct.pack("Q", 0))
            f.write(b)


class Model:
    def __init__(self, name):
        self.class_name = name
        self.model_dict = {}


def load_from(name, layers=9):
    dnnlib.tflib.init_tf()
    with open(name, 'rb') as f:
        m = pickle.load(f)

    Gs = m[2]

    model = Model('StyleGAN')

    def tensor(x, transpose=None):
        x = Gs.vars[x].eval()
        if transpose:
            x = np.transpose(x, transpose)
        return x

    for i in range(8):
        w = tensor('G_mapping/Dense%d/weight' % i, (1, 0))
        b = tensor('G_mapping/Dense%d/bias' % i)
        gain = np.sqrt(2.0)
        lrmul = 0.01
        std = gain / np.sqrt(w.shape[1]) * lrmul
        model.model_dict["mapping_block_%d_weight" % (i + 1)] = w * std
        model.model_dict["mapping_block_%d_bias" % (i + 1)] = b * lrmul

    model.model_dict["dlatent_avg"] = tensor('dlatent_avg')
    model.model_dict["block_0_const"] = tensor('G_synthesis/4x4/Const/const')

    rnd = np.random.RandomState(5)
    latents = rnd.randn(1, 512)
    model.model_dict["latents"] = latents.astype(np.float32)

    # custom trained rgb layers
    with open('rgbs.pkl', 'rb') as handle:
        rgbs = pickle.load(handle)

    for i in range(layers):
        j = layers - i - 1
        name = '%dx%d' % (2 ** (2 + i), 2 ** (2 + i))

        prefix = 'G_synthesis/%s' % name

        if i == 0:
            prefix_1 = '%s/Const' % prefix
            prefix_2 = '%s/Conv' % prefix
        else:
            prefix_1 = '%s/Conv0_up' % prefix
            prefix_2 = '%s/Conv1' % prefix

        model.model_dict["block_%d_noise_weight_1" % i] = tensor('%s/Noise/weight' % prefix_1)[None, :, None, None]
        model.model_dict["block_%d_noise_weight_2" % i] = tensor('%s/Noise/weight' % prefix_2)[None, :, None, None]

        if i != 0:
            if 2 ** (2 + i) >= 128:
                w = tensor('%s/weight' % prefix_1, (2, 3, 0, 1))
                gain = np.sqrt(2.0)
                std = gain / np.sqrt(np.prod([w.shape[0]] + list(w.shape[2:])))
                w = np.pad(w, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
                model.model_dict["block_%d_conv_1_weight" % i] = w * std
            else:
                w = tensor('%s/weight' % prefix_1, (3, 2, 0, 1))
                gain = np.sqrt(2.0)
                std = gain / np.sqrt(np.prod(w.shape[1:]))
                model.model_dict["block_%d_conv_1_weight" % i] = w * std

        gain = np.sqrt(2.0)
        w = tensor('%s/weight' % prefix_2, (3, 2, 0, 1))
        std = gain / np.sqrt(np.prod(w.shape[1:]))
        model.model_dict["block_%d_conv_2_weight" % i] = w * std

        model.model_dict["block_%d_bias_1" % i] = tensor('%s/bias' % prefix_1)[None, :, None, None]
        model.model_dict["block_%d_bias_2" % i] = tensor('%s/bias' % prefix_2)[None, :, None, None]

        gain = 1
        w = tensor('%s/StyleMod/weight' % prefix_1, (1, 0))
        std = gain / np.sqrt(np.prod(w.shape[1]))
        model.model_dict["block_%d_style_1_weight" % i] = w * std
        model.model_dict["block_%d_style_1_bias" % i] = tensor('%s/StyleMod/bias' % prefix_1)

        gain = 1
        w = tensor('%s/StyleMod/weight' % prefix_2, (1, 0))
        std = gain / np.sqrt(np.prod(w.shape[1]))
        model.model_dict["block_%d_style_2_weight" % i] = w * std
        model.model_dict["block_%d_style_2_bias" % i] = tensor('%s/StyleMod/bias' % prefix_2)

        gain = 1
        w = tensor('G_synthesis/ToRGB_lod%d/weight' % (j), (3, 2, 0, 1))
        std = gain / np.sqrt(np.prod(w.shape[1]))
        model.model_dict["block_%d_to_rgb_weight" % i] = w * std
        model.model_dict["block_%d_to_rgb_bias" % i] = tensor('G_synthesis/ToRGB_lod%d/bias' % (j))

        # ovveride to rgb layers with custom trained ones.
        model.model_dict["block_%d_to_rgb_weight" % i] = rgbs[i][0]
        model.model_dict["block_%d_to_rgb_bias" % i] = rgbs[i][1]

    return model #, Gs_


if __name__ == '__main__':
    model = load_from('karras2019stylegan-ffhq-1024x1024.pkl')
    write_blob(model)

    def write_h(x, *args):
        source_h.write(x % args)

    def write_cpp(x, *args):
        source_cpp.write(x % args)

    with open(model.class_name + ".cpp", "w") as source_cpp, open(model.class_name + ".h", "w") as source_h:
        write_h('#include "tensor4.h"' + '\n' * 3)
        write_cpp('#include "%s"' % (model.class_name + ".h") + '\n' * 3)

        write_h('struct %s\n{\n' % model.class_name)

        arguments = []

        for var_name, var in model.model_dict.items():
            var_type = 't4::tensor%df' % len(var.shape)
            decl_str = "\t%s %s;\n" % (var_type, var_name)
            write_h(decl_str)

        write_h('};' + '\n' * 3)

        declaration = '%s %sLoad(const char* filename)' % (model.class_name, model.class_name)
        write_h(declaration + ";\n\n")
        write_cpp(declaration + "\n{\n")
        write_cpp('\t%s ctx;\n', model.class_name)
        write_cpp('\tt4::model_dict dict = t4::load(filename);\n')

        for var_name, var in model.model_dict.items():
            string = "\tdict.load(ctx.%s, \"%s\", %s);\n" % (var_name, var_name, ', '.join([str(p) for p in var.shape]))
            write_cpp(string)

        write_cpp('\treturn ctx;\n}' + '\n' * 3)
