def parser(cfgfile):

    def _parse(l, i = 1):
        return l.split('=')[i].strip()

    with open(cfgfile, 'rb') as cfg:
        _file = cfg.readlines()

    lines = [line.decode() for line in _file]
    layers = list()
    for line in lines:
        line = line.strip()
        if '[' in line:
            layer = {'type': line}
            layers.append(layer)

        elif line == '':
            pass

        else:
            try:
                i = float(_parse(line))
                if i == int(i):
                    i = int(i)
                    layer[line.split('=')[0].strip()] = i
            except:
                try:
                    key = _parse(line, 0)
                    val = _parse(line, 1)
                    layer[key] = val
                except:
                    pass#print('Introduce parameter of layer like "key" = "value"') 
    return layers

def cfg_yielder(cfgfile):
    next_inp = None
    layers = parser(cfgfile)
    if layers[0].get('type') == '[input]':
        batch = layers[0].get('batch',None)
        inp_dim0 = layers[0].get('dim0')
        inp_dim1 = layers[0].get('dim1',None)
        inp_dim2 = layers[0].get('dim2',None)
        inp_init = [batch, inp_dim0, inp_dim1, inp_dim2]
        next_inp = inp_init
    for i, layer in enumerate(layers):
        if layer['type'] == '[input]':
            yield (inp_init, ['input', i])
        elif layer['type'] == '[connected]':
            outputs = layer.get('outputs')
            batchnorm = layer.get('batchnorm')
            activation = layer.get('activation')
            yield (inp_init, ['connected', i, next_inp, outputs, activation])
            next_inp = [batch,outputs]
            if batchnorm : yield (inp_init,['batchnorm',i,next_inp])
            if activation != 'linear': yield (inp_init ,[activation, i])
        elif layer['type'] == '[convolution]':
            filters = layer.get('filters')
            size = layer.get('size')
            pad = layer.get('pad')
            stride = layer.get('stride')
            batchnorm = layer.get('batchnorm')
            activation = layer.get('activation','linear')
            if pad: padding = size // 2
            yield (inp_init,['convolution', i, next_inp, filters, size, stride, padding, activation])
            w_ = (inp_dim0 + 2 * padding - size) // stride + 1
            h_ = (inp_dim1 + 2 * padding - size) // stride + 1
            inp_dim0, inp_dim1, inp_dim2 = w_, h_, filters
            next_inp = [batch,inp_dim0,inp_dim1,inp_dim2]
            if batchnorm : yield (inp_init,['batchnorm',i,next_inp])
            if activation != 'linear': yield (inp_init, [activation, i])
        elif layer['type'] == '[maxpool]':
            stride = layer.get('stride', 1)
            size = layer.get('size',stride)
            padding = layer.get('pad', (size-1)//2)
            yield (inp_init,['maxpool',i,size,stride,padding])
            w_ = (inp_dim0 + 2*padding - size) // stride + 1
            h_ = (inp_dim1 + 2*padding - size) // stride + 1
            inp_dim0, inp_dim1 = w_, h_
            next_inp = [batch,inp_dim0,inp_dim1,inp_dim2]
        elif layer['type'] == '[averagepool]':
            stride = layer.get('stride', 1)
            size = layer.get('size',stride)
            padding = layer.get('pad', (size-1)//2)
            yield (inp_init,['averagepool',i,size,stride,padding])
            w_ = (inp_dim0 + 2*padding - size) // stride + 1
            h_ = (inp_dim1 + 2*padding - size) // stride + 1
            inp_dim0, inp_dim1 = w_, h_
            next_inp = [batch,inp_dim0,inp_dim1,inp_dim2]
        elif layer['type'] == '[identity]':
            inp = layer.get('in')
            out = layer.get('out',None)
            yield (inp_init, ['identity',i,inp,out])
        elif layer['type'] == '[blockinit]':
            inp = layer.get('in')
            #length = layer.get('length')
            yield (inp_init, ['blockinit',i,inp])
        elif layer['type'] == '[blockend]':
            out = layer.get('out',None)
            yield (inp_init, ['blockend',i,out])
        else: pass

