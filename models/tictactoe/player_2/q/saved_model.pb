��

��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
�
q_network_2/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameq_network_2/dense_8/kernel
�
.q_network_2/dense_8/kernel/Read/ReadVariableOpReadVariableOpq_network_2/dense_8/kernel*
_output_shapes
:	�*
dtype0
�
q_network_2/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameq_network_2/dense_8/bias
�
,q_network_2/dense_8/bias/Read/ReadVariableOpReadVariableOpq_network_2/dense_8/bias*
_output_shapes	
:�*
dtype0
�
'q_network_2/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'q_network_2/batch_normalization_6/gamma
�
;q_network_2/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp'q_network_2/batch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
�
&q_network_2/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&q_network_2/batch_normalization_6/beta
�
:q_network_2/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp&q_network_2/batch_normalization_6/beta*
_output_shapes	
:�*
dtype0
�
-q_network_2/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-q_network_2/batch_normalization_6/moving_mean
�
Aq_network_2/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp-q_network_2/batch_normalization_6/moving_mean*
_output_shapes	
:�*
dtype0
�
1q_network_2/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31q_network_2/batch_normalization_6/moving_variance
�
Eq_network_2/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp1q_network_2/batch_normalization_6/moving_variance*
_output_shapes	
:�*
dtype0
�
q_network_2/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_nameq_network_2/dense_9/kernel
�
.q_network_2/dense_9/kernel/Read/ReadVariableOpReadVariableOpq_network_2/dense_9/kernel* 
_output_shapes
:
��*
dtype0
�
q_network_2/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameq_network_2/dense_9/bias
�
,q_network_2/dense_9/bias/Read/ReadVariableOpReadVariableOpq_network_2/dense_9/bias*
_output_shapes	
:�*
dtype0
�
'q_network_2/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'q_network_2/batch_normalization_7/gamma
�
;q_network_2/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp'q_network_2/batch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
&q_network_2/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&q_network_2/batch_normalization_7/beta
�
:q_network_2/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp&q_network_2/batch_normalization_7/beta*
_output_shapes	
:�*
dtype0
�
-q_network_2/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-q_network_2/batch_normalization_7/moving_mean
�
Aq_network_2/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp-q_network_2/batch_normalization_7/moving_mean*
_output_shapes	
:�*
dtype0
�
1q_network_2/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31q_network_2/batch_normalization_7/moving_variance
�
Eq_network_2/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp1q_network_2/batch_normalization_7/moving_variance*
_output_shapes	
:�*
dtype0
�
q_network_2/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nameq_network_2/dense_11/kernel
�
/q_network_2/dense_11/kernel/Read/ReadVariableOpReadVariableOpq_network_2/dense_11/kernel*
_output_shapes
:	�*
dtype0
�
q_network_2/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameq_network_2/dense_11/bias
�
-q_network_2/dense_11/bias/Read/ReadVariableOpReadVariableOpq_network_2/dense_11/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*� 
value� B�  B� 
�
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
layer_7
layer_8
		variables

regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
�
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-regularization_losses
.trainable_variables
/	keras_api

0	keras_api

1	keras_api
h

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
f
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
212
313
 
F
0
1
2
3
!4
"5
(6
)7
28
39
�
8non_trainable_variables

9layers
		variables
:layer_metrics

regularization_losses
;layer_regularization_losses
<metrics
trainable_variables
 
 
 
 
�
=non_trainable_variables

>layers
	variables
?layer_metrics
regularization_losses
@layer_regularization_losses
Ametrics
trainable_variables
YW
VARIABLE_VALUEq_network_2/dense_8/kernel)layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEq_network_2/dense_8/bias'layer_2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
Bnon_trainable_variables

Clayers
	variables
Dlayer_metrics
regularization_losses
Elayer_regularization_losses
Fmetrics
trainable_variables
 
ec
VARIABLE_VALUE'q_network_2/batch_normalization_6/gamma(layer_3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&q_network_2/batch_normalization_6/beta'layer_3/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE-q_network_2/batch_normalization_6/moving_mean.layer_3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE1q_network_2/batch_normalization_6/moving_variance2layer_3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
�
Gnon_trainable_variables

Hlayers
	variables
Ilayer_metrics
regularization_losses
Jlayer_regularization_losses
Kmetrics
trainable_variables
YW
VARIABLE_VALUEq_network_2/dense_9/kernel)layer_4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEq_network_2/dense_9/bias'layer_4/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
�
Lnon_trainable_variables

Mlayers
#	variables
Nlayer_metrics
$regularization_losses
Olayer_regularization_losses
Pmetrics
%trainable_variables
 
ec
VARIABLE_VALUE'q_network_2/batch_normalization_7/gamma(layer_5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&q_network_2/batch_normalization_7/beta'layer_5/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE-q_network_2/batch_normalization_7/moving_mean.layer_5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE1q_network_2/batch_normalization_7/moving_variance2layer_5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
*2
+3
 

(0
)1
�
Qnon_trainable_variables

Rlayers
,	variables
Slayer_metrics
-regularization_losses
Tlayer_regularization_losses
Umetrics
.trainable_variables
 
 
ZX
VARIABLE_VALUEq_network_2/dense_11/kernel)layer_8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEq_network_2/dense_11/bias'layer_8/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
�
Vnon_trainable_variables

Wlayers
4	variables
Xlayer_metrics
5regularization_losses
Ylayer_regularization_losses
Zmetrics
6trainable_variables

0
1
*2
+3
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

*0
+1
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_2Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2q_network_2/dense_8/kernelq_network_2/dense_8/bias-q_network_2/batch_normalization_6/moving_mean1q_network_2/batch_normalization_6/moving_variance&q_network_2/batch_normalization_6/beta'q_network_2/batch_normalization_6/gammaq_network_2/dense_9/kernelq_network_2/dense_9/bias-q_network_2/batch_normalization_7/moving_mean1q_network_2/batch_normalization_7/moving_variance&q_network_2/batch_normalization_7/beta'q_network_2/batch_normalization_7/gammaq_network_2/dense_11/kernelq_network_2/dense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_3322654
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.q_network_2/dense_8/kernel/Read/ReadVariableOp,q_network_2/dense_8/bias/Read/ReadVariableOp;q_network_2/batch_normalization_6/gamma/Read/ReadVariableOp:q_network_2/batch_normalization_6/beta/Read/ReadVariableOpAq_network_2/batch_normalization_6/moving_mean/Read/ReadVariableOpEq_network_2/batch_normalization_6/moving_variance/Read/ReadVariableOp.q_network_2/dense_9/kernel/Read/ReadVariableOp,q_network_2/dense_9/bias/Read/ReadVariableOp;q_network_2/batch_normalization_7/gamma/Read/ReadVariableOp:q_network_2/batch_normalization_7/beta/Read/ReadVariableOpAq_network_2/batch_normalization_7/moving_mean/Read/ReadVariableOpEq_network_2/batch_normalization_7/moving_variance/Read/ReadVariableOp/q_network_2/dense_11/kernel/Read/ReadVariableOp-q_network_2/dense_11/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__traced_save_3323174
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameq_network_2/dense_8/kernelq_network_2/dense_8/bias'q_network_2/batch_normalization_6/gamma&q_network_2/batch_normalization_6/beta-q_network_2/batch_normalization_6/moving_mean1q_network_2/batch_normalization_6/moving_varianceq_network_2/dense_9/kernelq_network_2/dense_9/bias'q_network_2/batch_normalization_7/gamma&q_network_2/batch_normalization_7/beta-q_network_2/batch_normalization_7/moving_mean1q_network_2/batch_normalization_7/moving_varianceq_network_2/dense_11/kernelq_network_2/dense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_3323226��
�
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3322272

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������	:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�h
�
"__inference__wrapped_model_3321980
input_1
input_26
2q_network_2_dense_8_matmul_readvariableop_resource7
3q_network_2_dense_8_biasadd_readvariableop_resourceB
>q_network_2_batch_normalization_6_cast_readvariableop_resourceD
@q_network_2_batch_normalization_6_cast_1_readvariableop_resourceD
@q_network_2_batch_normalization_6_cast_2_readvariableop_resourceD
@q_network_2_batch_normalization_6_cast_3_readvariableop_resource6
2q_network_2_dense_9_matmul_readvariableop_resource7
3q_network_2_dense_9_biasadd_readvariableop_resourceB
>q_network_2_batch_normalization_7_cast_readvariableop_resourceD
@q_network_2_batch_normalization_7_cast_1_readvariableop_resourceD
@q_network_2_batch_normalization_7_cast_2_readvariableop_resourceD
@q_network_2_batch_normalization_7_cast_3_readvariableop_resource7
3q_network_2_dense_11_matmul_readvariableop_resource8
4q_network_2_dense_11_biasadd_readvariableop_resource
identity��5q_network_2/batch_normalization_6/Cast/ReadVariableOp�7q_network_2/batch_normalization_6/Cast_1/ReadVariableOp�7q_network_2/batch_normalization_6/Cast_2/ReadVariableOp�7q_network_2/batch_normalization_6/Cast_3/ReadVariableOp�5q_network_2/batch_normalization_7/Cast/ReadVariableOp�7q_network_2/batch_normalization_7/Cast_1/ReadVariableOp�7q_network_2/batch_normalization_7/Cast_2/ReadVariableOp�7q_network_2/batch_normalization_7/Cast_3/ReadVariableOp�+q_network_2/dense_11/BiasAdd/ReadVariableOp�*q_network_2/dense_11/MatMul/ReadVariableOp�*q_network_2/dense_8/BiasAdd/ReadVariableOp�)q_network_2/dense_8/MatMul/ReadVariableOp�*q_network_2/dense_9/BiasAdd/ReadVariableOp�)q_network_2/dense_9/MatMul/ReadVariableOp�
%q_network_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%q_network_2/concatenate_2/concat/axis�
 q_network_2/concatenate_2/concatConcatV2input_1input_2.q_network_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2"
 q_network_2/concatenate_2/concat�
)q_network_2/dense_8/MatMul/ReadVariableOpReadVariableOp2q_network_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)q_network_2/dense_8/MatMul/ReadVariableOp�
q_network_2/dense_8/MatMulMatMul)q_network_2/concatenate_2/concat:output:01q_network_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_8/MatMul�
*q_network_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp3q_network_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*q_network_2/dense_8/BiasAdd/ReadVariableOp�
q_network_2/dense_8/BiasAddBiasAdd$q_network_2/dense_8/MatMul:product:02q_network_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_8/BiasAdd�
q_network_2/dense_8/ReluRelu$q_network_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_8/Relu�
5q_network_2/batch_normalization_6/Cast/ReadVariableOpReadVariableOp>q_network_2_batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype027
5q_network_2/batch_normalization_6/Cast/ReadVariableOp�
7q_network_2/batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_6/Cast_1/ReadVariableOp�
7q_network_2/batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_6/Cast_2/ReadVariableOp�
7q_network_2/batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_6/Cast_3/ReadVariableOp�
1q_network_2/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1q_network_2/batch_normalization_6/batchnorm/add/y�
/q_network_2/batch_normalization_6/batchnorm/addAddV2?q_network_2/batch_normalization_6/Cast_1/ReadVariableOp:value:0:q_network_2/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_6/batchnorm/add�
1q_network_2/batch_normalization_6/batchnorm/RsqrtRsqrt3q_network_2/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�23
1q_network_2/batch_normalization_6/batchnorm/Rsqrt�
/q_network_2/batch_normalization_6/batchnorm/mulMul5q_network_2/batch_normalization_6/batchnorm/Rsqrt:y:0?q_network_2/batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_6/batchnorm/mul�
1q_network_2/batch_normalization_6/batchnorm/mul_1Mul&q_network_2/dense_8/Relu:activations:03q_network_2/batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������23
1q_network_2/batch_normalization_6/batchnorm/mul_1�
1q_network_2/batch_normalization_6/batchnorm/mul_2Mul=q_network_2/batch_normalization_6/Cast/ReadVariableOp:value:03q_network_2/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�23
1q_network_2/batch_normalization_6/batchnorm/mul_2�
/q_network_2/batch_normalization_6/batchnorm/subSub?q_network_2/batch_normalization_6/Cast_2/ReadVariableOp:value:05q_network_2/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_6/batchnorm/sub�
1q_network_2/batch_normalization_6/batchnorm/add_1AddV25q_network_2/batch_normalization_6/batchnorm/mul_1:z:03q_network_2/batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������23
1q_network_2/batch_normalization_6/batchnorm/add_1�
)q_network_2/dense_9/MatMul/ReadVariableOpReadVariableOp2q_network_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02+
)q_network_2/dense_9/MatMul/ReadVariableOp�
q_network_2/dense_9/MatMulMatMul5q_network_2/batch_normalization_6/batchnorm/add_1:z:01q_network_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_9/MatMul�
*q_network_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp3q_network_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*q_network_2/dense_9/BiasAdd/ReadVariableOp�
q_network_2/dense_9/BiasAddBiasAdd$q_network_2/dense_9/MatMul:product:02q_network_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_9/BiasAdd�
q_network_2/dense_9/ReluRelu$q_network_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
q_network_2/dense_9/Relu�
5q_network_2/batch_normalization_7/Cast/ReadVariableOpReadVariableOp>q_network_2_batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype027
5q_network_2/batch_normalization_7/Cast/ReadVariableOp�
7q_network_2/batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_7/Cast_1/ReadVariableOp�
7q_network_2/batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_7/Cast_2/ReadVariableOp�
7q_network_2/batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp@q_network_2_batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7q_network_2/batch_normalization_7/Cast_3/ReadVariableOp�
1q_network_2/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:23
1q_network_2/batch_normalization_7/batchnorm/add/y�
/q_network_2/batch_normalization_7/batchnorm/addAddV2?q_network_2/batch_normalization_7/Cast_1/ReadVariableOp:value:0:q_network_2/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_7/batchnorm/add�
1q_network_2/batch_normalization_7/batchnorm/RsqrtRsqrt3q_network_2/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�23
1q_network_2/batch_normalization_7/batchnorm/Rsqrt�
/q_network_2/batch_normalization_7/batchnorm/mulMul5q_network_2/batch_normalization_7/batchnorm/Rsqrt:y:0?q_network_2/batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_7/batchnorm/mul�
1q_network_2/batch_normalization_7/batchnorm/mul_1Mul&q_network_2/dense_9/Relu:activations:03q_network_2/batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������23
1q_network_2/batch_normalization_7/batchnorm/mul_1�
1q_network_2/batch_normalization_7/batchnorm/mul_2Mul=q_network_2/batch_normalization_7/Cast/ReadVariableOp:value:03q_network_2/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�23
1q_network_2/batch_normalization_7/batchnorm/mul_2�
/q_network_2/batch_normalization_7/batchnorm/subSub?q_network_2/batch_normalization_7/Cast_2/ReadVariableOp:value:05q_network_2/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�21
/q_network_2/batch_normalization_7/batchnorm/sub�
1q_network_2/batch_normalization_7/batchnorm/add_1AddV25q_network_2/batch_normalization_7/batchnorm/mul_1:z:03q_network_2/batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������23
1q_network_2/batch_normalization_7/batchnorm/add_1�
*q_network_2/dense_11/MatMul/ReadVariableOpReadVariableOp3q_network_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*q_network_2/dense_11/MatMul/ReadVariableOp�
q_network_2/dense_11/MatMulMatMul5q_network_2/batch_normalization_7/batchnorm/add_1:z:02q_network_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
q_network_2/dense_11/MatMul�
+q_network_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp4q_network_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+q_network_2/dense_11/BiasAdd/ReadVariableOp�
q_network_2/dense_11/BiasAddBiasAdd%q_network_2/dense_11/MatMul:product:03q_network_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
q_network_2/dense_11/BiasAdd�
IdentityIdentity%q_network_2/dense_11/BiasAdd:output:06^q_network_2/batch_normalization_6/Cast/ReadVariableOp8^q_network_2/batch_normalization_6/Cast_1/ReadVariableOp8^q_network_2/batch_normalization_6/Cast_2/ReadVariableOp8^q_network_2/batch_normalization_6/Cast_3/ReadVariableOp6^q_network_2/batch_normalization_7/Cast/ReadVariableOp8^q_network_2/batch_normalization_7/Cast_1/ReadVariableOp8^q_network_2/batch_normalization_7/Cast_2/ReadVariableOp8^q_network_2/batch_normalization_7/Cast_3/ReadVariableOp,^q_network_2/dense_11/BiasAdd/ReadVariableOp+^q_network_2/dense_11/MatMul/ReadVariableOp+^q_network_2/dense_8/BiasAdd/ReadVariableOp*^q_network_2/dense_8/MatMul/ReadVariableOp+^q_network_2/dense_9/BiasAdd/ReadVariableOp*^q_network_2/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2n
5q_network_2/batch_normalization_6/Cast/ReadVariableOp5q_network_2/batch_normalization_6/Cast/ReadVariableOp2r
7q_network_2/batch_normalization_6/Cast_1/ReadVariableOp7q_network_2/batch_normalization_6/Cast_1/ReadVariableOp2r
7q_network_2/batch_normalization_6/Cast_2/ReadVariableOp7q_network_2/batch_normalization_6/Cast_2/ReadVariableOp2r
7q_network_2/batch_normalization_6/Cast_3/ReadVariableOp7q_network_2/batch_normalization_6/Cast_3/ReadVariableOp2n
5q_network_2/batch_normalization_7/Cast/ReadVariableOp5q_network_2/batch_normalization_7/Cast/ReadVariableOp2r
7q_network_2/batch_normalization_7/Cast_1/ReadVariableOp7q_network_2/batch_normalization_7/Cast_1/ReadVariableOp2r
7q_network_2/batch_normalization_7/Cast_2/ReadVariableOp7q_network_2/batch_normalization_7/Cast_2/ReadVariableOp2r
7q_network_2/batch_normalization_7/Cast_3/ReadVariableOp7q_network_2/batch_normalization_7/Cast_3/ReadVariableOp2Z
+q_network_2/dense_11/BiasAdd/ReadVariableOp+q_network_2/dense_11/BiasAdd/ReadVariableOp2X
*q_network_2/dense_11/MatMul/ReadVariableOp*q_network_2/dense_11/MatMul/ReadVariableOp2X
*q_network_2/dense_8/BiasAdd/ReadVariableOp*q_network_2/dense_8/BiasAdd/ReadVariableOp2V
)q_network_2/dense_8/MatMul/ReadVariableOp)q_network_2/dense_8/MatMul/ReadVariableOp2X
*q_network_2/dense_9/BiasAdd/ReadVariableOp*q_network_2/dense_9/BiasAdd/ReadVariableOp2V
)q_network_2/dense_9/MatMul/ReadVariableOp)q_network_2/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3322879
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������	:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�

�
-__inference_q_network_2_layer_call_fn_3322618
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_q_network_2_layer_call_and_return_conditional_losses_33225872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3322249

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_dense_8_layer_call_and_return_conditional_losses_3322896

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322109

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
)__inference_dense_8_layer_call_fn_3322905

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_33222922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_6_layer_call_fn_3322987

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33221092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_dense_11_layer_call_fn_3323108

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_33224152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_dense_9_layer_call_and_return_conditional_losses_3322354

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322076

inputs
assignmovingavg_3322051
assignmovingavg_1_3322057 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322051*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3322051*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322051*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322051*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3322051AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322051*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322057*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3322057*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322057*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322057*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3322057AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322057*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323063

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�U
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322804
inputs_0
inputs_1*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource6
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource8
4batch_normalization_6_cast_2_readvariableop_resource8
4batch_normalization_6_cast_3_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource6
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource8
4batch_normalization_7_cast_2_readvariableop_resource8
4batch_normalization_7_cast_3_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOpx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis�
concatenate_2/concatConcatV2inputs_0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate_2/concat�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulconcatenate_2/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_8/Relu�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd�
IdentityIdentitydense_11/BiasAdd:output:0*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�/
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322941

inputs
assignmovingavg_3322916
assignmovingavg_1_3322922 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322916*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3322916*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322916*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322916*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3322916AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322916*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322922*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3322922*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322922*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322922*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3322922AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322922*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322961

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_7_layer_call_fn_3323089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3322216

inputs
assignmovingavg_3322191
assignmovingavg_1_3322197 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322191*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3322191*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322191*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3322191*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3322191AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3322191*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322197*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3322197*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322197*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3322197*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3322197AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3322197*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
-__inference_q_network_2_layer_call_fn_3322838
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_q_network_2_layer_call_and_return_conditional_losses_33225142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�	
�
D__inference_dense_8_layer_call_and_return_conditional_losses_3322292

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_6_layer_call_fn_3322974

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33220762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322514

inputs
inputs_1
dense_8_3322480
dense_8_3322482!
batch_normalization_6_3322485!
batch_normalization_6_3322487!
batch_normalization_6_3322489!
batch_normalization_6_3322491
dense_9_3322494
dense_9_3322496!
batch_normalization_7_3322499!
batch_normalization_7_3322501!
batch_normalization_7_3322503!
batch_normalization_7_3322505
dense_11_3322508
dense_11_3322510
identity��-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_33222722
concatenate_2/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_3322480dense_8_3322482*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_33222922!
dense_8/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_3322485batch_normalization_6_3322487batch_normalization_6_3322489batch_normalization_6_3322491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33220762/
-batch_normalization_6/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_9_3322494dense_9_3322496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_33223542!
dense_9/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_3322499batch_normalization_7_3322501batch_normalization_7_3322503batch_normalization_7_3322505*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222162/
-batch_normalization_7/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_11_3322508dense_11_3322510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_33224152"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
[
/__inference_concatenate_2_layer_call_fn_3322885
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_33222722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������	:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�+
�
 __inference__traced_save_3323174
file_prefix9
5savev2_q_network_2_dense_8_kernel_read_readvariableop7
3savev2_q_network_2_dense_8_bias_read_readvariableopF
Bsavev2_q_network_2_batch_normalization_6_gamma_read_readvariableopE
Asavev2_q_network_2_batch_normalization_6_beta_read_readvariableopL
Hsavev2_q_network_2_batch_normalization_6_moving_mean_read_readvariableopP
Lsavev2_q_network_2_batch_normalization_6_moving_variance_read_readvariableop9
5savev2_q_network_2_dense_9_kernel_read_readvariableop7
3savev2_q_network_2_dense_9_bias_read_readvariableopF
Bsavev2_q_network_2_batch_normalization_7_gamma_read_readvariableopE
Asavev2_q_network_2_batch_normalization_7_beta_read_readvariableopL
Hsavev2_q_network_2_batch_normalization_7_moving_mean_read_readvariableopP
Lsavev2_q_network_2_batch_normalization_7_moving_variance_read_readvariableop:
6savev2_q_network_2_dense_11_kernel_read_readvariableop8
4savev2_q_network_2_dense_11_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer_3/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer_3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer_3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer_4/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_4/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer_5/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer_5/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer_5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer_5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer_8/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_q_network_2_dense_8_kernel_read_readvariableop3savev2_q_network_2_dense_8_bias_read_readvariableopBsavev2_q_network_2_batch_normalization_6_gamma_read_readvariableopAsavev2_q_network_2_batch_normalization_6_beta_read_readvariableopHsavev2_q_network_2_batch_normalization_6_moving_mean_read_readvariableopLsavev2_q_network_2_batch_normalization_6_moving_variance_read_readvariableop5savev2_q_network_2_dense_9_kernel_read_readvariableop3savev2_q_network_2_dense_9_bias_read_readvariableopBsavev2_q_network_2_batch_normalization_7_gamma_read_readvariableopAsavev2_q_network_2_batch_normalization_7_beta_read_readvariableopHsavev2_q_network_2_batch_normalization_7_moving_mean_read_readvariableopLsavev2_q_network_2_batch_normalization_7_moving_variance_read_readvariableop6savev2_q_network_2_dense_11_kernel_read_readvariableop4savev2_q_network_2_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapest
r: :	�:�:�:�:�:�:
��:�:�:�:�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�

�
%__inference_signature_wrapper_3322654
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_33219802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�
�
7__inference_batch_normalization_7_layer_call_fn_3323076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_11_layer_call_and_return_conditional_losses_3322415

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
)__inference_dense_9_layer_call_fn_3323007

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_33223542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
-__inference_q_network_2_layer_call_fn_3322872
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_q_network_2_layer_call_and_return_conditional_losses_33225872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�	
�
D__inference_dense_9_layer_call_and_return_conditional_losses_3322998

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322471
input_1
input_2
dense_8_3322437
dense_8_3322439!
batch_normalization_6_3322442!
batch_normalization_6_3322444!
batch_normalization_6_3322446!
batch_normalization_6_3322448
dense_9_3322451
dense_9_3322453!
batch_normalization_7_3322456!
batch_normalization_7_3322458!
batch_normalization_7_3322460!
batch_normalization_7_3322462
dense_11_3322465
dense_11_3322467
identity��-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
concatenate_2/PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_33222722
concatenate_2/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_3322437dense_8_3322439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_33222922!
dense_8/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_3322442batch_normalization_6_3322444batch_normalization_6_3322446batch_normalization_6_3322448*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33221092/
-batch_normalization_6/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_9_3322451dense_9_3322453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_33223542!
dense_9/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_3322456batch_normalization_7_3322458batch_normalization_7_3322460batch_normalization_7_3322462*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222492/
-batch_normalization_7/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_11_3322465dense_11_3322467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_33224152"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�$
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322432
input_1
input_2
dense_8_3322303
dense_8_3322305!
batch_normalization_6_3322334!
batch_normalization_6_3322336!
batch_normalization_6_3322338!
batch_normalization_6_3322340
dense_9_3322365
dense_9_3322367!
batch_normalization_7_3322396!
batch_normalization_7_3322398!
batch_normalization_7_3322400!
batch_normalization_7_3322402
dense_11_3322426
dense_11_3322428
identity��-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
concatenate_2/PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_33222722
concatenate_2/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_3322303dense_8_3322305*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_33222922!
dense_8/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_3322334batch_normalization_6_3322336batch_normalization_6_3322338batch_normalization_6_3322340*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33220762/
-batch_normalization_6/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_9_3322365dense_9_3322367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_33223542!
dense_9/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_3322396batch_normalization_7_3322398batch_normalization_7_3322400batch_normalization_7_3322402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222162/
-batch_normalization_7/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_11_3322426dense_11_3322428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_33224152"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322745
inputs_0
inputs_1*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource1
-batch_normalization_6_assignmovingavg_33226753
/batch_normalization_6_assignmovingavg_1_33226816
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource1
-batch_normalization_7_assignmovingavg_33227143
/batch_normalization_7_assignmovingavg_1_33227206
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOpx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis�
concatenate_2/concatConcatV2inputs_0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate_2/concat�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulconcatenate_2/concat:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_8/Relu�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeandense_8/Relu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_8/Relu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/3322675*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_6_assignmovingavg_3322675*
_output_shapes	
:�*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/3322675*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/3322675*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/mul�
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_6_assignmovingavg_3322675-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/3322675*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/3322681*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_6_assignmovingavg_1_3322681*
_output_shapes	
:�*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/3322681*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/3322681*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/mul�
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_6_assignmovingavg_1_3322681/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/3322681*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_9/Relu:activations:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:03batch_normalization_7/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg/3322714*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_7_assignmovingavg_3322714*
_output_shapes	
:�*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg/3322714*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg/3322714*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/mul�
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_7_assignmovingavg_3322714-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg/3322714*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_7/AssignMovingAvg_1/3322720*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_7_assignmovingavg_1_3322720*
_output_shapes	
:�*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_7/AssignMovingAvg_1/3322720*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_7/AssignMovingAvg_1/3322720*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/mul�
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_7_assignmovingavg_1_3322720/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_7/AssignMovingAvg_1/3322720*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/Relu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMul)batch_normalization_7/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdd�
IdentityIdentitydense_11/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�/
�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323043

inputs
assignmovingavg_3323018
assignmovingavg_1_3323024 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3323018*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_3323018*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3323018*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/3323018*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_3323018AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/3323018*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3323024*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_3323024*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3323024*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/3323024*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_3323024AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/3323024*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
-__inference_q_network_2_layer_call_fn_3322545
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_q_network_2_layer_call_and_return_conditional_losses_33225142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������	
!
_user_specified_name	input_2
�@
�	
#__inference__traced_restore_3323226
file_prefix/
+assignvariableop_q_network_2_dense_8_kernel/
+assignvariableop_1_q_network_2_dense_8_bias>
:assignvariableop_2_q_network_2_batch_normalization_6_gamma=
9assignvariableop_3_q_network_2_batch_normalization_6_betaD
@assignvariableop_4_q_network_2_batch_normalization_6_moving_meanH
Dassignvariableop_5_q_network_2_batch_normalization_6_moving_variance1
-assignvariableop_6_q_network_2_dense_9_kernel/
+assignvariableop_7_q_network_2_dense_9_bias>
:assignvariableop_8_q_network_2_batch_normalization_7_gamma=
9assignvariableop_9_q_network_2_batch_normalization_7_betaE
Aassignvariableop_10_q_network_2_batch_normalization_7_moving_meanI
Eassignvariableop_11_q_network_2_batch_normalization_7_moving_variance3
/assignvariableop_12_q_network_2_dense_11_kernel1
-assignvariableop_13_q_network_2_dense_11_bias
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer_3/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer_3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer_3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer_4/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_4/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer_5/gamma/.ATTRIBUTES/VARIABLE_VALUEB'layer_5/beta/.ATTRIBUTES/VARIABLE_VALUEB.layer_5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2layer_5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)layer_8/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer_8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp+assignvariableop_q_network_2_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_q_network_2_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp:assignvariableop_2_q_network_2_batch_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_q_network_2_batch_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp@assignvariableop_4_q_network_2_batch_normalization_6_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpDassignvariableop_5_q_network_2_batch_normalization_6_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_q_network_2_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_q_network_2_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp:assignvariableop_8_q_network_2_batch_normalization_7_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_q_network_2_batch_normalization_7_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpAassignvariableop_10_q_network_2_batch_normalization_7_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpEassignvariableop_11_q_network_2_batch_normalization_7_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_q_network_2_dense_11_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp-assignvariableop_13_q_network_2_dense_11_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14�
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
E__inference_dense_11_layer_call_and_return_conditional_losses_3323099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322587

inputs
inputs_1
dense_8_3322553
dense_8_3322555!
batch_normalization_6_3322558!
batch_normalization_6_3322560!
batch_normalization_6_3322562!
batch_normalization_6_3322564
dense_9_3322567
dense_9_3322569!
batch_normalization_7_3322572!
batch_normalization_7_3322574!
batch_normalization_7_3322576!
batch_normalization_7_3322578
dense_11_3322581
dense_11_3322583
identity��-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
concatenate_2/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_33222722
concatenate_2/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_8_3322553dense_8_3322555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_33222922!
dense_8/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_6_3322558batch_normalization_6_3322560batch_normalization_6_3322562batch_normalization_6_3322564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_33221092/
-batch_normalization_6/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_9_3322567dense_9_3322569*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_33223542!
dense_9/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_7_3322572batch_normalization_7_3322574batch_normalization_7_3322576batch_normalization_7_3322578*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_33222492/
-batch_normalization_7/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0dense_11_3322581dense_11_3322583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_33224152"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������	::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������	
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������
;
input_20
serving_default_input_2:0���������	<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
layer_7
layer_8
		variables

regularization_losses
trainable_variables
	keras_api

signatures
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "QNetwork", "name": "q_network_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}}
�
	variables
regularization_losses
trainable_variables
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 9]}]}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 27}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27]}}
�	
axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
 	keras_api
b__call__
*c&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
�

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
d__call__
*e&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
�	
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-regularization_losses
.trainable_variables
/	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
0	keras_api"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {}}}}
�
1	keras_api"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
�

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h__call__
*i&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
212
313"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
!4
"5
(6
)7
28
39"
trackable_list_wrapper
�
8non_trainable_variables

9layers
		variables
:layer_metrics

regularization_losses
;layer_regularization_losses
<metrics
trainable_variables
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
=non_trainable_variables

>layers
	variables
?layer_metrics
regularization_losses
@layer_regularization_losses
Ametrics
trainable_variables
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
-:+	�2q_network_2/dense_8/kernel
':%�2q_network_2/dense_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
	variables
Dlayer_metrics
regularization_losses
Elayer_regularization_losses
Fmetrics
trainable_variables
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4�2'q_network_2/batch_normalization_6/gamma
5:3�2&q_network_2/batch_normalization_6/beta
>:<� (2-q_network_2/batch_normalization_6/moving_mean
B:@� (21q_network_2/batch_normalization_6/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
	variables
Ilayer_metrics
regularization_losses
Jlayer_regularization_losses
Kmetrics
trainable_variables
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
.:,
��2q_network_2/dense_9/kernel
':%�2q_network_2/dense_9/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
#	variables
Nlayer_metrics
$regularization_losses
Olayer_regularization_losses
Pmetrics
%trainable_variables
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4�2'q_network_2/batch_normalization_7/gamma
5:3�2&q_network_2/batch_normalization_7/beta
>:<� (2-q_network_2/batch_normalization_7/moving_mean
B:@� (21q_network_2/batch_normalization_7/moving_variance
<
(0
)1
*2
+3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
Qnon_trainable_variables

Rlayers
,	variables
Slayer_metrics
-regularization_losses
Tlayer_regularization_losses
Umetrics
.trainable_variables
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
.:,	�2q_network_2/dense_11/kernel
':%2q_network_2/dense_11/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
4	variables
Xlayer_metrics
5regularization_losses
Ylayer_regularization_losses
Zmetrics
6trainable_variables
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
<
0
1
*2
+3"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
-__inference_q_network_2_layer_call_fn_3322838
-__inference_q_network_2_layer_call_fn_3322618
-__inference_q_network_2_layer_call_fn_3322872
-__inference_q_network_2_layer_call_fn_3322545�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_3321980�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������
!�
input_2���������	
�2�
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322745
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322432
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322804
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322471�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_concatenate_2_layer_call_fn_3322885�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3322879�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_8_layer_call_fn_3322905�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_8_layer_call_and_return_conditional_losses_3322896�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_6_layer_call_fn_3322974
7__inference_batch_normalization_6_layer_call_fn_3322987�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322941
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322961�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dense_9_layer_call_fn_3323007�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_9_layer_call_and_return_conditional_losses_3322998�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_7_layer_call_fn_3323089
7__inference_batch_normalization_7_layer_call_fn_3323076�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323063
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323043�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_11_layer_call_fn_3323108�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_11_layer_call_and_return_conditional_losses_3323099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_3322654input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_3321980�!"*+)(23X�U
N�K
I�F
!�
input_1���������
!�
input_2���������	
� "3�0
.
output_1"�
output_1����������
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322941d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3322961d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
7__inference_batch_normalization_6_layer_call_fn_3322974W4�1
*�'
!�
inputs����������
p
� "������������
7__inference_batch_normalization_6_layer_call_fn_3322987W4�1
*�'
!�
inputs����������
p 
� "������������
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323043d*+)(4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3323063d*+)(4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
7__inference_batch_normalization_7_layer_call_fn_3323076W*+)(4�1
*�'
!�
inputs����������
p
� "������������
7__inference_batch_normalization_7_layer_call_fn_3323089W*+)(4�1
*�'
!�
inputs����������
p 
� "������������
J__inference_concatenate_2_layer_call_and_return_conditional_losses_3322879�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������	
� "%�"
�
0���������
� �
/__inference_concatenate_2_layer_call_fn_3322885vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������	
� "�����������
E__inference_dense_11_layer_call_and_return_conditional_losses_3323099]230�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_11_layer_call_fn_3323108P230�-
&�#
!�
inputs����������
� "�����������
D__inference_dense_8_layer_call_and_return_conditional_losses_3322896]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� }
)__inference_dense_8_layer_call_fn_3322905P/�,
%�"
 �
inputs���������
� "������������
D__inference_dense_9_layer_call_and_return_conditional_losses_3322998^!"0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_9_layer_call_fn_3323007Q!"0�-
&�#
!�
inputs����������
� "������������
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322432�!"*+)(23`�]
V�S
I�F
!�
input_1���������
!�
input_2���������	
p

 
� "%�"
�
0���������
� �
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322471�!"*+)(23`�]
V�S
I�F
!�
input_1���������
!�
input_2���������	
p 

 
� "%�"
�
0���������
� �
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322745�!"*+)(23b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������	
p

 
� "%�"
�
0���������
� �
H__inference_q_network_2_layer_call_and_return_conditional_losses_3322804�!"*+)(23b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������	
p 

 
� "%�"
�
0���������
� �
-__inference_q_network_2_layer_call_fn_3322545�!"*+)(23`�]
V�S
I�F
!�
input_1���������
!�
input_2���������	
p

 
� "�����������
-__inference_q_network_2_layer_call_fn_3322618�!"*+)(23`�]
V�S
I�F
!�
input_1���������
!�
input_2���������	
p 

 
� "�����������
-__inference_q_network_2_layer_call_fn_3322838�!"*+)(23b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������	
p

 
� "�����������
-__inference_q_network_2_layer_call_fn_3322872�!"*+)(23b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������	
p 

 
� "�����������
%__inference_signature_wrapper_3322654�!"*+)(23i�f
� 
_�\
,
input_1!�
input_1���������
,
input_2!�
input_2���������	"3�0
.
output_1"�
output_1���������