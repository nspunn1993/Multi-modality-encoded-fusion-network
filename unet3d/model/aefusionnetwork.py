from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, BatchNormalization, Deconvolution3D, MaxPooling3D, Multiply
from keras.engine import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from .unet import create_convolution_block, concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from ..metrics import weighted_dice_coefficient_loss, reg_loss, similarity_value, real_similarity_value, tloss,weighted_dice_coefficient,iou_loss,comb_loss

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, batch_normalization=True)


def aefusionnetwork_model(input_shape=(1, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid", multi_gpu_flag=False):
    
    fi_1 = Input(input_shape)
    fi_2 = Input(input_shape)
    fi_3 = Input(input_shape)
    fi_4 = Input(input_shape)
    
    fconv1_1 = create_inception_module(fi_1, 1)
    fconv1_2 = create_inception_module(fi_2, 1)
    fconv1_3 = create_inception_module(fi_3, 1)
    fconv1_4 = create_inception_module(fi_4, 1)
    
    fconv1_concat_1 = concatenate([fconv1_1,fconv1_2], axis=1)
    fconv1_concat_2 = concatenate([fconv1_3,fconv1_4], axis=1)
    
    fconv2_1 = create_inception_module(fconv1_concat_1, 2)
    fconv2_2 = create_inception_module(fconv1_concat_2, 2)
    
    fconv2_concat_1 = concatenate([fconv2_1,fconv2_2], axis=1)
    
    fconv3_1 = create_inception_module(fconv2_concat_1, 4)
    
    inputs = fconv3_1
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)
        if current_layer is inputs:
            #in_conv = create_inception_module(current_layer, n_level_filters, concat=False)
            in_conv = concatenate([current_layer,fconv1_concat_1,fconv1_concat_2],axis=1)
        else:
            in_conv = create_inception_module(current_layer, n_level_filters, strides=(2, 2, 2))
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = get_up_convolution(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
            #output_layer = get_up_convolution(output_layer, 3)
    '''
    concat0_1 = Add()([output_layer,inputs])
    dconv1_1 = create_inception_module(concat0_1, 2)
    dconv1_2 = create_inception_module(dconv1_1, 1)
    add_dconv1 = Add()([concat0_1,dconv1_2])
    dconv1_3 = create_inception_module(add_dconv1, 1)
    
    dconcat1_1 = Multiply()([dconv1_3,concat0_1])
    dconv1_4 = create_inception_module(dconcat1_1, 2)
    dconv1_5 = create_inception_module(dconv1_4, 1)
    add_dconv2 = Add()([dconcat1_1,dconv1_5])
    dconv1_6 = create_inception_module(add_dconv2, 1)
    
    dconcat1_2 = Multiply()([dconv1_6,concat0_1])
    dconv1_7 = create_inception_module(dconcat1_2, 2)
    dconv1_8 = create_inception_module(dconv1_7, 1)
    add_dconv3 = Add()([dconcat1_2,dconv1_8])
    dconv1_9 = create_inception_module(add_dconv3, 1)
    
    activation_block1 = Activation(activation_name)(dconv1_3)
    activation_block2 = Activation(activation_name)(dconv1_6)
    activation_block3 = Activation(activation_name)(dconv1_9)
    '''
    
    #concat0_1 = concatenate([output_layer,inputs], axis=1)
    #dconv1_1 = create_inception_module(output_layer, 1)
    #dconcat1_1 = concatenate([dconv1_1,concat0_1],axis=1)
    #dconv1_2 = create_inception_module(dconv1_1, 1)
    #concat1_2 = concatenate([dconv1_2,dconv1_1],axis=1)
    #dconv1_3 = create_inception_module(dconv1_2, 1)
    
    #activation_block1 = Activation(activation_name)(dconv1_1)
    #activation_block2 = Activation(activation_name)(dconv1_2)
    #activation_block3 = Activation(activation_name)(dconv1_3)
    
    activation_block4 = Activation(activation_name)(output_layer)
    
    #activation_concat = Add()([activation_block1,activation_block2,activation_block3])
    #dconv1_4 = create_inception_module(activation_concat, 3)
    #activation_block4 = Activation(activation_name)(dconv1_4)
    
    model = Model(inputs=[fi_1,fi_2,fi_3,fi_4], outputs=[activation_block4])
    
    if multi_gpu_flag:
        model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=[comb_loss], metrics=[dice_coefficient])
    return model


def create_localization_module(input_layer, n_filters, dropout_rate=0.3, data_format="channels_first"):
    local_module = create_inception_module(input_layer, n_filters, upsample=True)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(local_module)
    return dropout


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), dropout_rate=0.3, data_format="channels_first"):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_inception_module(up_sample, n_filters, upsample=True)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution)
    return dropout

def get_up_convolution(input_layer,n_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    
    return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides, kernel_initializer='he_normal', activation='elu') (input_layer)

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_inception_module(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_inception_module(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def create_inception_module(input_layer, n_filters, strides=(1,1,1), concat=True, upsample=False):
    in_conv_1 = create_convolution_block(input_layer, n_filters, kernel=(1,1,1), strides=strides, instance_normalization=True, batch_normalization=False)
    in_conv_2 = create_convolution_block(input_layer, n_filters, kernel=(3,3,3), strides=strides, instance_normalization=True, batch_normalization=False)
    in_conv_3 = create_convolution_block(input_layer, n_filters, kernel=(5,5,5), strides=strides, instance_normalization=True, batch_normalization=False)
    #mpool_1 = MaxPooling3D((2, 2, 2), padding='same', strides=strides) (input_layer)
    if concat:
        in_concat = concatenate([in_conv_1,in_conv_2,in_conv_3], axis=1)
    else:
        in_concat = Add()([in_conv_1,in_conv_2,in_conv_3])
    in_conv = create_convolution_block(in_concat, n_filters, kernel=(1,1,1), instance_normalization=True)
    if not upsample:
        mpool_1 = MaxPooling3D((2, 2, 2), padding='same', strides=(1,1,1)) (in_conv)
    else:
        mpool_1 = in_conv
    return mpool_1
    