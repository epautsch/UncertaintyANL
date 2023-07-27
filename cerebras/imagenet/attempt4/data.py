from ImageNetLoader import ImageNetLoader


def input_fn_train(params):
    input_params = params['train_input']
    batch_size = input_params.get('batch_size')

    train_dl = ImageNetLoader(split='train', batch_size=batch_size)
    return train_dl


def input_fn_eval(params):
    input_params = params['val_input']
    batch_size = input_params.get('batch_size')

    val_dl = ImageNetLoader(split='val', batch_size=batch_size)
    return val_dl
