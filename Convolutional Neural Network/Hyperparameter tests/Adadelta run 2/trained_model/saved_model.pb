۸

??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02unknown8??
?
cnn__model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecnn__model/conv2d/kernel
?
,cnn__model/conv2d/kernel/Read/ReadVariableOpReadVariableOpcnn__model/conv2d/kernel*&
_output_shapes
:*
dtype0
?
cnn__model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecnn__model/conv2d/bias
}
*cnn__model/conv2d/bias/Read/ReadVariableOpReadVariableOpcnn__model/conv2d/bias*
_output_shapes
:*
dtype0
?
cnn__model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namecnn__model/conv2d_1/kernel
?
.cnn__model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpcnn__model/conv2d_1/kernel*&
_output_shapes
: *
dtype0
?
cnn__model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namecnn__model/conv2d_1/bias
?
,cnn__model/conv2d_1/bias/Read/ReadVariableOpReadVariableOpcnn__model/conv2d_1/bias*
_output_shapes
: *
dtype0
?
cnn__model/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_namecnn__model/conv2d_2/kernel
?
.cnn__model/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpcnn__model/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
?
cnn__model/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namecnn__model/conv2d_2/bias
?
,cnn__model/conv2d_2/bias/Read/ReadVariableOpReadVariableOpcnn__model/conv2d_2/bias*
_output_shapes
:@*
dtype0
?
cnn__model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_namecnn__model/dense/kernel
?
+cnn__model/dense/kernel/Read/ReadVariableOpReadVariableOpcnn__model/dense/kernel* 
_output_shapes
:
??*
dtype0
?
cnn__model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_namecnn__model/dense/bias
|
)cnn__model/dense/bias/Read/ReadVariableOpReadVariableOpcnn__model/dense/bias*
_output_shapes	
:?*
dtype0
?
cnn__model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
**
shared_namecnn__model/dense_1/kernel
?
-cnn__model/dense_1/kernel/Read/ReadVariableOpReadVariableOpcnn__model/dense_1/kernel*
_output_shapes
:	?
*
dtype0
?
cnn__model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namecnn__model/dense_1/bias

+cnn__model/dense_1/bias/Read/ReadVariableOpReadVariableOpcnn__model/dense_1/bias*
_output_shapes
:
*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
?
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
,Adadelta/cnn__model/conv2d/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adadelta/cnn__model/conv2d/kernel/accum_grad
?
@Adadelta/cnn__model/conv2d/kernel/accum_grad/Read/ReadVariableOpReadVariableOp,Adadelta/cnn__model/conv2d/kernel/accum_grad*&
_output_shapes
:*
dtype0
?
*Adadelta/cnn__model/conv2d/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adadelta/cnn__model/conv2d/bias/accum_grad
?
>Adadelta/cnn__model/conv2d/bias/accum_grad/Read/ReadVariableOpReadVariableOp*Adadelta/cnn__model/conv2d/bias/accum_grad*
_output_shapes
:*
dtype0
?
.Adadelta/cnn__model/conv2d_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adadelta/cnn__model/conv2d_1/kernel/accum_grad
?
BAdadelta/cnn__model/conv2d_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp.Adadelta/cnn__model/conv2d_1/kernel/accum_grad*&
_output_shapes
: *
dtype0
?
,Adadelta/cnn__model/conv2d_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adadelta/cnn__model/conv2d_1/bias/accum_grad
?
@Adadelta/cnn__model/conv2d_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp,Adadelta/cnn__model/conv2d_1/bias/accum_grad*
_output_shapes
: *
dtype0
?
.Adadelta/cnn__model/conv2d_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*?
shared_name0.Adadelta/cnn__model/conv2d_2/kernel/accum_grad
?
BAdadelta/cnn__model/conv2d_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp.Adadelta/cnn__model/conv2d_2/kernel/accum_grad*&
_output_shapes
: @*
dtype0
?
,Adadelta/cnn__model/conv2d_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adadelta/cnn__model/conv2d_2/bias/accum_grad
?
@Adadelta/cnn__model/conv2d_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp,Adadelta/cnn__model/conv2d_2/bias/accum_grad*
_output_shapes
:@*
dtype0
?
+Adadelta/cnn__model/dense/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*<
shared_name-+Adadelta/cnn__model/dense/kernel/accum_grad
?
?Adadelta/cnn__model/dense/kernel/accum_grad/Read/ReadVariableOpReadVariableOp+Adadelta/cnn__model/dense/kernel/accum_grad* 
_output_shapes
:
??*
dtype0
?
)Adadelta/cnn__model/dense/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)Adadelta/cnn__model/dense/bias/accum_grad
?
=Adadelta/cnn__model/dense/bias/accum_grad/Read/ReadVariableOpReadVariableOp)Adadelta/cnn__model/dense/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
-Adadelta/cnn__model/dense_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*>
shared_name/-Adadelta/cnn__model/dense_1/kernel/accum_grad
?
AAdadelta/cnn__model/dense_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp-Adadelta/cnn__model/dense_1/kernel/accum_grad*
_output_shapes
:	?
*
dtype0
?
+Adadelta/cnn__model/dense_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*<
shared_name-+Adadelta/cnn__model/dense_1/bias/accum_grad
?
?Adadelta/cnn__model/dense_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp+Adadelta/cnn__model/dense_1/bias/accum_grad*
_output_shapes
:
*
dtype0
?
+Adadelta/cnn__model/conv2d/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adadelta/cnn__model/conv2d/kernel/accum_var
?
?Adadelta/cnn__model/conv2d/kernel/accum_var/Read/ReadVariableOpReadVariableOp+Adadelta/cnn__model/conv2d/kernel/accum_var*&
_output_shapes
:*
dtype0
?
)Adadelta/cnn__model/conv2d/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adadelta/cnn__model/conv2d/bias/accum_var
?
=Adadelta/cnn__model/conv2d/bias/accum_var/Read/ReadVariableOpReadVariableOp)Adadelta/cnn__model/conv2d/bias/accum_var*
_output_shapes
:*
dtype0
?
-Adadelta/cnn__model/conv2d_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adadelta/cnn__model/conv2d_1/kernel/accum_var
?
AAdadelta/cnn__model/conv2d_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp-Adadelta/cnn__model/conv2d_1/kernel/accum_var*&
_output_shapes
: *
dtype0
?
+Adadelta/cnn__model/conv2d_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adadelta/cnn__model/conv2d_1/bias/accum_var
?
?Adadelta/cnn__model/conv2d_1/bias/accum_var/Read/ReadVariableOpReadVariableOp+Adadelta/cnn__model/conv2d_1/bias/accum_var*
_output_shapes
: *
dtype0
?
-Adadelta/cnn__model/conv2d_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*>
shared_name/-Adadelta/cnn__model/conv2d_2/kernel/accum_var
?
AAdadelta/cnn__model/conv2d_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp-Adadelta/cnn__model/conv2d_2/kernel/accum_var*&
_output_shapes
: @*
dtype0
?
+Adadelta/cnn__model/conv2d_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+Adadelta/cnn__model/conv2d_2/bias/accum_var
?
?Adadelta/cnn__model/conv2d_2/bias/accum_var/Read/ReadVariableOpReadVariableOp+Adadelta/cnn__model/conv2d_2/bias/accum_var*
_output_shapes
:@*
dtype0
?
*Adadelta/cnn__model/dense/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adadelta/cnn__model/dense/kernel/accum_var
?
>Adadelta/cnn__model/dense/kernel/accum_var/Read/ReadVariableOpReadVariableOp*Adadelta/cnn__model/dense/kernel/accum_var* 
_output_shapes
:
??*
dtype0
?
(Adadelta/cnn__model/dense/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(Adadelta/cnn__model/dense/bias/accum_var
?
<Adadelta/cnn__model/dense/bias/accum_var/Read/ReadVariableOpReadVariableOp(Adadelta/cnn__model/dense/bias/accum_var*
_output_shapes	
:?*
dtype0
?
,Adadelta/cnn__model/dense_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*=
shared_name.,Adadelta/cnn__model/dense_1/kernel/accum_var
?
@Adadelta/cnn__model/dense_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp,Adadelta/cnn__model/dense_1/kernel/accum_var*
_output_shapes
:	?
*
dtype0
?
*Adadelta/cnn__model/dense_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adadelta/cnn__model/dense_1/bias/accum_var
?
>Adadelta/cnn__model/dense_1/bias/accum_var/Read/ReadVariableOpReadVariableOp*Adadelta/cnn__model/dense_1/bias/accum_var*
_output_shapes
:
*
dtype0

NoOpNoOp
?s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?s
value?sB?s B?s
?
conv_layer_1
dropout_layer_1
pool_layer_1
dropout_layer_2
conv_layer_2
dropout_layer_3
pool_layer_2
dropout_layer_4
	conv_layer_3

dropout_layer_5
flatten_layer
dropout_layer_6
hidden_layer
dropout_layer_7
output_layer
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
?
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
?
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 
?
#2_self_saveable_object_factories
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses* 
?

:kernel
;bias
#<_self_saveable_object_factories
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H_random_generator
I__call__
*J&call_and_return_all_conditional_losses* 
?
#K_self_saveable_object_factories
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
?
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
X__call__
*Y&call_and_return_all_conditional_losses* 
?

Zkernel
[bias
#\_self_saveable_object_factories
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses*
?
#c_self_saveable_object_factories
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h_random_generator
i__call__
*j&call_and_return_all_conditional_losses* 
?
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
?
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w_random_generator
x__call__
*y&call_and_return_all_conditional_losses* 
?

zkernel
{bias
#|_self_saveable_object_factories
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter

?decay
?learning_rate
?rho
accum_grad?
accum_grad?:
accum_grad?;
accum_grad?Z
accum_grad?[
accum_grad?z
accum_grad?{
accum_grad??
accum_grad??
accum_grad?	accum_var?	accum_var?:	accum_var?;	accum_var?Z	accum_var?[	accum_var?z	accum_var?{	accum_var??	accum_var??	accum_var?*

?serving_default* 
* 
L
0
1
:2
;3
Z4
[5
z6
{7
?8
?9*
L
0
1
:2
;3
Z4
[5
z6
{7
?8
?9*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEcnn__model/conv2d/kernel.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEcnn__model/conv2d/bias,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 
* 
* 
* 
b\
VARIABLE_VALUEcnn__model/conv2d_1/kernel.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcnn__model/conv2d_1/bias,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

:0
;1*

:0
;1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 
* 
* 
* 
b\
VARIABLE_VALUEcnn__model/conv2d_2/kernel.conv_layer_3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEcnn__model/conv2d_2/bias,conv_layer_3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Z0
[1*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEcnn__model/dense/kernel.hidden_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcnn__model/dense/bias,hidden_layer/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

z0
{1*

z0
{1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
a[
VARIABLE_VALUEcnn__model/dense_1/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcnn__model/dense_1/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
PJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUE,Adadelta/cnn__model/conv2d/kernel/accum_gradSconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adadelta/cnn__model/conv2d/bias/accum_gradQconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adadelta/cnn__model/conv2d_1/kernel/accum_gradSconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adadelta/cnn__model/conv2d_1/bias/accum_gradQconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adadelta/cnn__model/conv2d_2/kernel/accum_gradSconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adadelta/cnn__model/conv2d_2/bias/accum_gradQconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adadelta/cnn__model/dense/kernel/accum_gradShidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adadelta/cnn__model/dense/bias/accum_gradQhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adadelta/cnn__model/dense_1/kernel/accum_gradSoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adadelta/cnn__model/dense_1/bias/accum_gradQoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adadelta/cnn__model/conv2d/kernel/accum_varRconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adadelta/cnn__model/conv2d/bias/accum_varPconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adadelta/cnn__model/conv2d_1/kernel/accum_varRconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adadelta/cnn__model/conv2d_1/bias/accum_varPconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adadelta/cnn__model/conv2d_2/kernel/accum_varRconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adadelta/cnn__model/conv2d_2/bias/accum_varPconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adadelta/cnn__model/dense/kernel/accum_varRhidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE(Adadelta/cnn__model/dense/bias/accum_varPhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adadelta/cnn__model/dense_1/kernel/accum_varRoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adadelta/cnn__model/dense_1/bias/accum_varPoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_args_0Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0cnn__model/conv2d/kernelcnn__model/conv2d/biascnn__model/conv2d_1/kernelcnn__model/conv2d_1/biascnn__model/conv2d_2/kernelcnn__model/conv2d_2/biascnn__model/dense/kernelcnn__model/dense/biascnn__model/dense_1/kernelcnn__model/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_693098
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,cnn__model/conv2d/kernel/Read/ReadVariableOp*cnn__model/conv2d/bias/Read/ReadVariableOp.cnn__model/conv2d_1/kernel/Read/ReadVariableOp,cnn__model/conv2d_1/bias/Read/ReadVariableOp.cnn__model/conv2d_2/kernel/Read/ReadVariableOp,cnn__model/conv2d_2/bias/Read/ReadVariableOp+cnn__model/dense/kernel/Read/ReadVariableOp)cnn__model/dense/bias/Read/ReadVariableOp-cnn__model/dense_1/kernel/Read/ReadVariableOp+cnn__model/dense_1/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp@Adadelta/cnn__model/conv2d/kernel/accum_grad/Read/ReadVariableOp>Adadelta/cnn__model/conv2d/bias/accum_grad/Read/ReadVariableOpBAdadelta/cnn__model/conv2d_1/kernel/accum_grad/Read/ReadVariableOp@Adadelta/cnn__model/conv2d_1/bias/accum_grad/Read/ReadVariableOpBAdadelta/cnn__model/conv2d_2/kernel/accum_grad/Read/ReadVariableOp@Adadelta/cnn__model/conv2d_2/bias/accum_grad/Read/ReadVariableOp?Adadelta/cnn__model/dense/kernel/accum_grad/Read/ReadVariableOp=Adadelta/cnn__model/dense/bias/accum_grad/Read/ReadVariableOpAAdadelta/cnn__model/dense_1/kernel/accum_grad/Read/ReadVariableOp?Adadelta/cnn__model/dense_1/bias/accum_grad/Read/ReadVariableOp?Adadelta/cnn__model/conv2d/kernel/accum_var/Read/ReadVariableOp=Adadelta/cnn__model/conv2d/bias/accum_var/Read/ReadVariableOpAAdadelta/cnn__model/conv2d_1/kernel/accum_var/Read/ReadVariableOp?Adadelta/cnn__model/conv2d_1/bias/accum_var/Read/ReadVariableOpAAdadelta/cnn__model/conv2d_2/kernel/accum_var/Read/ReadVariableOp?Adadelta/cnn__model/conv2d_2/bias/accum_var/Read/ReadVariableOp>Adadelta/cnn__model/dense/kernel/accum_var/Read/ReadVariableOp<Adadelta/cnn__model/dense/bias/accum_var/Read/ReadVariableOp@Adadelta/cnn__model/dense_1/kernel/accum_var/Read/ReadVariableOp>Adadelta/cnn__model/dense_1/bias/accum_var/Read/ReadVariableOpConst*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_693279
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn__model/conv2d/kernelcnn__model/conv2d/biascnn__model/conv2d_1/kernelcnn__model/conv2d_1/biascnn__model/conv2d_2/kernelcnn__model/conv2d_2/biascnn__model/dense/kernelcnn__model/dense/biascnn__model/dense_1/kernelcnn__model/dense_1/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1,Adadelta/cnn__model/conv2d/kernel/accum_grad*Adadelta/cnn__model/conv2d/bias/accum_grad.Adadelta/cnn__model/conv2d_1/kernel/accum_grad,Adadelta/cnn__model/conv2d_1/bias/accum_grad.Adadelta/cnn__model/conv2d_2/kernel/accum_grad,Adadelta/cnn__model/conv2d_2/bias/accum_grad+Adadelta/cnn__model/dense/kernel/accum_grad)Adadelta/cnn__model/dense/bias/accum_grad-Adadelta/cnn__model/dense_1/kernel/accum_grad+Adadelta/cnn__model/dense_1/bias/accum_grad+Adadelta/cnn__model/conv2d/kernel/accum_var)Adadelta/cnn__model/conv2d/bias/accum_var-Adadelta/cnn__model/conv2d_1/kernel/accum_var+Adadelta/cnn__model/conv2d_1/bias/accum_var-Adadelta/cnn__model/conv2d_2/kernel/accum_var+Adadelta/cnn__model/conv2d_2/bias/accum_var*Adadelta/cnn__model/dense/kernel/accum_var(Adadelta/cnn__model/dense/bias/accum_var,Adadelta/cnn__model/dense_1/kernel/accum_var*Adadelta/cnn__model/dense_1/bias/accum_var*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_693403??
?

a
B__inference_dropout_2_layer_call_and_return_conditional_losses_289

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????

 C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????

 *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

 w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

 q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????

 a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????

 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

 :W S
/
_output_shapes
:?????????

 
 
_user_specified_nameinputs
?

?
)__inference_cnn__model_layer_call_fn_1072

input_data!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn__model_layer_call_and_return_conditional_losses_1057`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
??
?
"__inference__traced_restore_693403
file_prefixC
)assignvariableop_cnn__model_conv2d_kernel:7
)assignvariableop_1_cnn__model_conv2d_bias:G
-assignvariableop_2_cnn__model_conv2d_1_kernel: 9
+assignvariableop_3_cnn__model_conv2d_1_bias: G
-assignvariableop_4_cnn__model_conv2d_2_kernel: @9
+assignvariableop_5_cnn__model_conv2d_2_bias:@>
*assignvariableop_6_cnn__model_dense_kernel:
??7
(assignvariableop_7_cnn__model_dense_bias:	??
,assignvariableop_8_cnn__model_dense_1_kernel:	?
8
*assignvariableop_9_cnn__model_dense_1_bias:
+
!assignvariableop_10_adadelta_iter:	 ,
"assignvariableop_11_adadelta_decay: 4
*assignvariableop_12_adadelta_learning_rate: *
 assignvariableop_13_adadelta_rho: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: Z
@assignvariableop_18_adadelta_cnn__model_conv2d_kernel_accum_grad:L
>assignvariableop_19_adadelta_cnn__model_conv2d_bias_accum_grad:\
Bassignvariableop_20_adadelta_cnn__model_conv2d_1_kernel_accum_grad: N
@assignvariableop_21_adadelta_cnn__model_conv2d_1_bias_accum_grad: \
Bassignvariableop_22_adadelta_cnn__model_conv2d_2_kernel_accum_grad: @N
@assignvariableop_23_adadelta_cnn__model_conv2d_2_bias_accum_grad:@S
?assignvariableop_24_adadelta_cnn__model_dense_kernel_accum_grad:
??L
=assignvariableop_25_adadelta_cnn__model_dense_bias_accum_grad:	?T
Aassignvariableop_26_adadelta_cnn__model_dense_1_kernel_accum_grad:	?
M
?assignvariableop_27_adadelta_cnn__model_dense_1_bias_accum_grad:
Y
?assignvariableop_28_adadelta_cnn__model_conv2d_kernel_accum_var:K
=assignvariableop_29_adadelta_cnn__model_conv2d_bias_accum_var:[
Aassignvariableop_30_adadelta_cnn__model_conv2d_1_kernel_accum_var: M
?assignvariableop_31_adadelta_cnn__model_conv2d_1_bias_accum_var: [
Aassignvariableop_32_adadelta_cnn__model_conv2d_2_kernel_accum_var: @M
?assignvariableop_33_adadelta_cnn__model_conv2d_2_bias_accum_var:@R
>assignvariableop_34_adadelta_cnn__model_dense_kernel_accum_var:
??K
<assignvariableop_35_adadelta_cnn__model_dense_bias_accum_var:	?S
@assignvariableop_36_adadelta_cnn__model_dense_1_kernel_accum_var:	?
L
>assignvariableop_37_adadelta_cnn__model_dense_1_bias_accum_var:

identity_39??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_3/bias/.ATTRIBUTES/VARIABLE_VALUEB.hidden_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,hidden_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBShidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRhidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp)assignvariableop_cnn__model_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_cnn__model_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_cnn__model_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_cnn__model_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_cnn__model_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_cnn__model_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_cnn__model_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_cnn__model_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_cnn__model_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_cnn__model_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_adadelta_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_adadelta_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adadelta_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_adadelta_rhoIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adadelta_cnn__model_conv2d_kernel_accum_gradIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adadelta_cnn__model_conv2d_bias_accum_gradIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_adadelta_cnn__model_conv2d_1_kernel_accum_gradIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adadelta_cnn__model_conv2d_1_bias_accum_gradIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adadelta_cnn__model_conv2d_2_kernel_accum_gradIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adadelta_cnn__model_conv2d_2_bias_accum_gradIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp?assignvariableop_24_adadelta_cnn__model_dense_kernel_accum_gradIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp=assignvariableop_25_adadelta_cnn__model_dense_bias_accum_gradIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adadelta_cnn__model_dense_1_kernel_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp?assignvariableop_27_adadelta_cnn__model_dense_1_bias_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp?assignvariableop_28_adadelta_cnn__model_conv2d_kernel_accum_varIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adadelta_cnn__model_conv2d_bias_accum_varIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpAassignvariableop_30_adadelta_cnn__model_conv2d_1_kernel_accum_varIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adadelta_cnn__model_conv2d_1_bias_accum_varIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpAassignvariableop_32_adadelta_cnn__model_conv2d_2_kernel_accum_varIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adadelta_cnn__model_conv2d_2_bias_accum_varIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adadelta_cnn__model_dense_kernel_accum_varIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adadelta_cnn__model_dense_bias_accum_varIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp@assignvariableop_36_adadelta_cnn__model_dense_1_kernel_accum_varIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adadelta_cnn__model_dense_1_bias_accum_varIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_39IdentityIdentity_38:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
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
?n
?
C__inference_cnn__model_layer_call_and_return_conditional_losses_277

input_data?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?
5
'dense_1_biasadd_readvariableop_resource:

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D
input_data$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMulconv2d/Relu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????^
dropout/dropout/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
max_pooling2d/MaxPoolMaxPooldropout/dropout/Mul_1:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMulmax_pooling2d/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????e
dropout_1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMulconv2d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????

 b
dropout_2/dropout/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????

 *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????

 ?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????

 ?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????

 ?
max_pooling2d_1/MaxPoolMaxPooldropout_2/dropout/Mul_1:z:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_3/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:????????? g
dropout_3/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? ?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? ?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? ?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_2/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_4/dropout/MulMulconv2d_2/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@b
dropout_4/dropout/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_5/dropout/MulMulflatten/Reshape:output:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:??????????_
dropout_5/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMuldropout_5/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_6/dropout/MulMuldense/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*(
_output_shapes
:??????????_
dropout_6/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_1/MatMulMatMuldropout_6/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_961

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_693142

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_683

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????

 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_693137

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_693129?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_693107

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_dense_1_layer_call_and_return_conditional_losses_977

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
)__inference_cnn__model_layer_call_fn_1087
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn__model_layer_call_and_return_conditional_losses_1057`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?#
?
D__inference_cnn__model_layer_call_and_return_conditional_losses_1109

input_data'
conv2d_484726:
conv2d_484728:)
conv2d_1_484744: 
conv2d_1_484746: )
conv2d_2_484762: @
conv2d_2_484764:@ 
dense_484787:
??
dense_484789:	?!
dense_1_484804:	?

dense_1_484806:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCall
input_dataconv2d_484726conv2d_484728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_591?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_961?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_484744conv2d_1_484746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_683?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_830?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_484762conv2d_2_484764*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_813?
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_873?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_484787dense_484789*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_701?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_484804dense_1_484806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_977?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?
\
@__inference_flatten_layer_call_and_return_conditional_losses_873

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_693129

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

a
B__inference_dropout_1_layer_call_and_return_conditional_losses_392

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
>__inference_dense_layer_call_and_return_conditional_losses_701

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

a
B__inference_dropout_3_layer_call_and_return_conditional_losses_929

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
)__inference_cnn__model_layer_call_fn_1124

input_data!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn__model_layer_call_and_return_conditional_losses_1109`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?

?
?__inference_conv2d_layer_call_and_return_conditional_losses_591

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_693120

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?X
?
__inference__traced_save_693279
file_prefix7
3savev2_cnn__model_conv2d_kernel_read_readvariableop5
1savev2_cnn__model_conv2d_bias_read_readvariableop9
5savev2_cnn__model_conv2d_1_kernel_read_readvariableop7
3savev2_cnn__model_conv2d_1_bias_read_readvariableop9
5savev2_cnn__model_conv2d_2_kernel_read_readvariableop7
3savev2_cnn__model_conv2d_2_bias_read_readvariableop6
2savev2_cnn__model_dense_kernel_read_readvariableop4
0savev2_cnn__model_dense_bias_read_readvariableop8
4savev2_cnn__model_dense_1_kernel_read_readvariableop6
2savev2_cnn__model_dense_1_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopK
Gsavev2_adadelta_cnn__model_conv2d_kernel_accum_grad_read_readvariableopI
Esavev2_adadelta_cnn__model_conv2d_bias_accum_grad_read_readvariableopM
Isavev2_adadelta_cnn__model_conv2d_1_kernel_accum_grad_read_readvariableopK
Gsavev2_adadelta_cnn__model_conv2d_1_bias_accum_grad_read_readvariableopM
Isavev2_adadelta_cnn__model_conv2d_2_kernel_accum_grad_read_readvariableopK
Gsavev2_adadelta_cnn__model_conv2d_2_bias_accum_grad_read_readvariableopJ
Fsavev2_adadelta_cnn__model_dense_kernel_accum_grad_read_readvariableopH
Dsavev2_adadelta_cnn__model_dense_bias_accum_grad_read_readvariableopL
Hsavev2_adadelta_cnn__model_dense_1_kernel_accum_grad_read_readvariableopJ
Fsavev2_adadelta_cnn__model_dense_1_bias_accum_grad_read_readvariableopJ
Fsavev2_adadelta_cnn__model_conv2d_kernel_accum_var_read_readvariableopH
Dsavev2_adadelta_cnn__model_conv2d_bias_accum_var_read_readvariableopL
Hsavev2_adadelta_cnn__model_conv2d_1_kernel_accum_var_read_readvariableopJ
Fsavev2_adadelta_cnn__model_conv2d_1_bias_accum_var_read_readvariableopL
Hsavev2_adadelta_cnn__model_conv2d_2_kernel_accum_var_read_readvariableopJ
Fsavev2_adadelta_cnn__model_conv2d_2_bias_accum_var_read_readvariableopI
Esavev2_adadelta_cnn__model_dense_kernel_accum_var_read_readvariableopG
Csavev2_adadelta_cnn__model_dense_bias_accum_var_read_readvariableopK
Gsavev2_adadelta_cnn__model_dense_1_kernel_accum_var_read_readvariableopI
Esavev2_adadelta_cnn__model_dense_1_bias_accum_var_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_3/bias/.ATTRIBUTES/VARIABLE_VALUEB.hidden_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,hidden_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBShidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBSoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBQoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRconv_layer_3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPconv_layer_3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRhidden_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPhidden_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBRoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBPoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_cnn__model_conv2d_kernel_read_readvariableop1savev2_cnn__model_conv2d_bias_read_readvariableop5savev2_cnn__model_conv2d_1_kernel_read_readvariableop3savev2_cnn__model_conv2d_1_bias_read_readvariableop5savev2_cnn__model_conv2d_2_kernel_read_readvariableop3savev2_cnn__model_conv2d_2_bias_read_readvariableop2savev2_cnn__model_dense_kernel_read_readvariableop0savev2_cnn__model_dense_bias_read_readvariableop4savev2_cnn__model_dense_1_kernel_read_readvariableop2savev2_cnn__model_dense_1_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopGsavev2_adadelta_cnn__model_conv2d_kernel_accum_grad_read_readvariableopEsavev2_adadelta_cnn__model_conv2d_bias_accum_grad_read_readvariableopIsavev2_adadelta_cnn__model_conv2d_1_kernel_accum_grad_read_readvariableopGsavev2_adadelta_cnn__model_conv2d_1_bias_accum_grad_read_readvariableopIsavev2_adadelta_cnn__model_conv2d_2_kernel_accum_grad_read_readvariableopGsavev2_adadelta_cnn__model_conv2d_2_bias_accum_grad_read_readvariableopFsavev2_adadelta_cnn__model_dense_kernel_accum_grad_read_readvariableopDsavev2_adadelta_cnn__model_dense_bias_accum_grad_read_readvariableopHsavev2_adadelta_cnn__model_dense_1_kernel_accum_grad_read_readvariableopFsavev2_adadelta_cnn__model_dense_1_bias_accum_grad_read_readvariableopFsavev2_adadelta_cnn__model_conv2d_kernel_accum_var_read_readvariableopDsavev2_adadelta_cnn__model_conv2d_bias_accum_var_read_readvariableopHsavev2_adadelta_cnn__model_conv2d_1_kernel_accum_var_read_readvariableopFsavev2_adadelta_cnn__model_conv2d_1_bias_accum_var_read_readvariableopHsavev2_adadelta_cnn__model_conv2d_2_kernel_accum_var_read_readvariableopFsavev2_adadelta_cnn__model_conv2d_2_bias_accum_var_read_readvariableopEsavev2_adadelta_cnn__model_dense_kernel_accum_var_read_readvariableopCsavev2_adadelta_cnn__model_dense_bias_accum_var_read_readvariableopGsavev2_adadelta_cnn__model_dense_1_kernel_accum_var_read_readvariableopEsavev2_adadelta_cnn__model_dense_1_bias_accum_var_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : @:@:
??:?:	?
:
: : : : : : : : ::: : : @:@:
??:?:	?
:
::: : : @:@:
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: :  

_output_shapes
: :,!(
&
_output_shapes
: @: "

_output_shapes
:@:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:%%!

_output_shapes
:	?
: &

_output_shapes
:
:'

_output_shapes
: 
?	
a
B__inference_dropout_5_layer_call_and_return_conditional_losses_162

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_693067

args_0+
cnn__model_693045:
cnn__model_693047:+
cnn__model_693049: 
cnn__model_693051: +
cnn__model_693053: @
cnn__model_693055:@%
cnn__model_693057:
?? 
cnn__model_693059:	?$
cnn__model_693061:	?

cnn__model_693063:

identity??"cnn__model/StatefulPartitionedCall?
"cnn__model/StatefulPartitionedCallStatefulPartitionedCallargs_0cnn__model_693045cnn__model_693047cnn__model_693049cnn__model_693051cnn__model_693053cnn__model_693055cnn__model_693057cnn__model_693059cnn__model_693061cnn__model_693063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_693044z
IdentityIdentity+cnn__model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
k
NoOpNoOp#^cnn__model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2H
"cnn__model/StatefulPartitionedCall"cnn__model/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameargs_0
?

?
$__inference_signature_wrapper_693098

args_0!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_693067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameargs_0
?

?
)__inference_restored_function_body_693044

input_data!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_cnn__model_layer_call_and_return_conditional_losses_359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?#
?
C__inference_cnn__model_layer_call_and_return_conditional_losses_999
input_1'
conv2d_485209:
conv2d_485211:)
conv2d_1_485215: 
conv2d_1_485217: )
conv2d_2_485221: @
conv2d_2_485223:@ 
dense_485227:
??
dense_485229:	?!
dense_1_485232:	?

dense_1_485234:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_485209conv2d_485211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_591?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_961?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_485215conv2d_1_485217*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_683?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_830?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_485221conv2d_2_485223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_813?
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_873?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_485227dense_485229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_701?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_485232dense_1_485234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_977?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
J
.__inference_max_pooling2d_layer_call_fn_693115

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_693107?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_2_layer_call_and_return_conditional_losses_813

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

_
@__inference_dropout_layer_call_and_return_conditional_losses_946

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
)__inference_cnn__model_layer_call_fn_1139
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn__model_layer_call_and_return_conditional_losses_1109`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?A
?
D__inference_cnn__model_layer_call_and_return_conditional_losses_1028
input_1'
conv2d_485241:
conv2d_485243:)
conv2d_1_485249: 
conv2d_1_485251: )
conv2d_2_485257: @
conv2d_2_485259:@ 
dense_485265:
??
dense_485267:	?!
dense_1_485271:	?

dense_1_485273:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_485241conv2d_485243*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_591?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_946?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_961?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_392?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_485249conv2d_1_485251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_683?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_289?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_830?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_929?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_2_485257conv2d_2_485259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_813?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_4_layer_call_and_return_conditional_losses_536?
flatten/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_873?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_5_layer_call_and_return_conditional_losses_162?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_485265dense_485267*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_701?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_6_layer_call_and_return_conditional_losses_553?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_1_485271dense_1_485273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_977?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?A
?
D__inference_cnn__model_layer_call_and_return_conditional_losses_1057

input_data'
conv2d_484919:
conv2d_484921:)
conv2d_1_484953: 
conv2d_1_484955: )
conv2d_2_484987: @
conv2d_2_484989:@ 
dense_485021:
??
dense_485023:	?!
dense_1_485040:	?

dense_1_485042:

identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCall
input_dataconv2d_484919conv2d_484921*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_591?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_946?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_961?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_392?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_484953conv2d_1_484955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_683?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_289?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_830?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_929?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_2_484987conv2d_2_484989*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_813?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_4_layer_call_and_return_conditional_losses_536?
flatten/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_873?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_5_layer_call_and_return_conditional_losses_162?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_485021dense_485023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_701?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_6_layer_call_and_return_conditional_losses_553?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_1_485040dense_1_485042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_977?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?2
?
C__inference_cnn__model_layer_call_and_return_conditional_losses_359

input_data?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?
5
'dense_1_biasadd_readvariableop_resource:

identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D
input_data$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 ?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
/
_output_shapes
:?????????  
$
_user_specified_name
input_data
?
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_830

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
a
B__inference_dropout_6_layer_call_and_return_conditional_losses_553

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

a
B__inference_dropout_4_layer_call_and_return_conditional_losses_536

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
args_07
serving_default_args_0:0?????????  <
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
conv_layer_1
dropout_layer_1
pool_layer_1
dropout_layer_2
conv_layer_2
dropout_layer_3
pool_layer_2
dropout_layer_4
	conv_layer_3

dropout_layer_5
flatten_layer
dropout_layer_6
hidden_layer
dropout_layer_7
output_layer
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_model
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
?
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#2_self_saveable_object_factories
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
#<_self_saveable_object_factories
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H_random_generator
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#K_self_saveable_object_factories
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Zkernel
[bias
#\_self_saveable_object_factories
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#c_self_saveable_object_factories
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h_random_generator
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w_random_generator
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?

zkernel
{bias
#|_self_saveable_object_factories
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter

?decay
?learning_rate
?rho
accum_grad?
accum_grad?:
accum_grad?;
accum_grad?Z
accum_grad?[
accum_grad?z
accum_grad?{
accum_grad??
accum_grad??
accum_grad?	accum_var?	accum_var?:	accum_var?;	accum_var?Z	accum_var?[	accum_var?z	accum_var?{	accum_var??	accum_var??	accum_var?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
h
0
1
:2
;3
Z4
[5
z6
{7
?8
?9"
trackable_list_wrapper
h
0
1
:2
;3
Z4
[5
z6
{7
?8
?9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_cnn__model_layer_call_fn_1139
)__inference_cnn__model_layer_call_fn_1124
)__inference_cnn__model_layer_call_fn_1072
)__inference_cnn__model_layer_call_fn_1087?
???
FullArgSpec%
args?
j
input_data

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_cnn__model_layer_call_and_return_conditional_losses_359
C__inference_cnn__model_layer_call_and_return_conditional_losses_277
C__inference_cnn__model_layer_call_and_return_conditional_losses_999
D__inference_cnn__model_layer_call_and_return_conditional_losses_1028?
???
FullArgSpec%
args?
j
input_data

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_693067args_0"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2:02cnn__model/conv2d/kernel
$:"2cnn__model/conv2d/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_max_pooling2d_layer_call_fn_693115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_693120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
4:2 2cnn__model/conv2d_1/kernel
&:$ 2cnn__model/conv2d_1/bias
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_max_pooling2d_1_layer_call_fn_693137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_693142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
4:2 @2cnn__model/conv2d_2/kernel
&:$@2cnn__model/conv2d_2/bias
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
+:)
??2cnn__model/dense/kernel
$:"?2cnn__model/dense/bias
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
,:*	?
2cnn__model/dense_1/kernel
%:#
2cnn__model/dense_1/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
?B?
$__inference_signature_wrapper_693098args_0"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
?0
?1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
D:B2,Adadelta/cnn__model/conv2d/kernel/accum_grad
6:42*Adadelta/cnn__model/conv2d/bias/accum_grad
F:D 2.Adadelta/cnn__model/conv2d_1/kernel/accum_grad
8:6 2,Adadelta/cnn__model/conv2d_1/bias/accum_grad
F:D @2.Adadelta/cnn__model/conv2d_2/kernel/accum_grad
8:6@2,Adadelta/cnn__model/conv2d_2/bias/accum_grad
=:;
??2+Adadelta/cnn__model/dense/kernel/accum_grad
6:4?2)Adadelta/cnn__model/dense/bias/accum_grad
>:<	?
2-Adadelta/cnn__model/dense_1/kernel/accum_grad
7:5
2+Adadelta/cnn__model/dense_1/bias/accum_grad
C:A2+Adadelta/cnn__model/conv2d/kernel/accum_var
5:32)Adadelta/cnn__model/conv2d/bias/accum_var
E:C 2-Adadelta/cnn__model/conv2d_1/kernel/accum_var
7:5 2+Adadelta/cnn__model/conv2d_1/bias/accum_var
E:C @2-Adadelta/cnn__model/conv2d_2/kernel/accum_var
7:5@2+Adadelta/cnn__model/conv2d_2/bias/accum_var
<::
??2*Adadelta/cnn__model/dense/kernel/accum_var
5:3?2(Adadelta/cnn__model/dense/bias/accum_var
=:;	?
2,Adadelta/cnn__model/dense_1/kernel/accum_var
6:4
2*Adadelta/cnn__model/dense_1/bias/accum_var?
!__inference__wrapped_model_693067|:;Z[z{??7?4
-?*
(?%
args_0?????????  
? "3?0
.
output_1"?
output_1?????????
?
D__inference_cnn__model_layer_call_and_return_conditional_losses_1028s:;Z[z{??<?9
2?/
)?&
input_1?????????  
p
? "%?"
?
0?????????

? ?
C__inference_cnn__model_layer_call_and_return_conditional_losses_277v:;Z[z{????<
5?2
,?)

input_data?????????  
p
? "%?"
?
0?????????

? ?
C__inference_cnn__model_layer_call_and_return_conditional_losses_359v:;Z[z{????<
5?2
,?)

input_data?????????  
p 
? "%?"
?
0?????????

? ?
C__inference_cnn__model_layer_call_and_return_conditional_losses_999s:;Z[z{??<?9
2?/
)?&
input_1?????????  
p 
? "%?"
?
0?????????

? ?
)__inference_cnn__model_layer_call_fn_1072i:;Z[z{????<
5?2
,?)

input_data?????????  
p
? "??????????
?
)__inference_cnn__model_layer_call_fn_1087f:;Z[z{??<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
)__inference_cnn__model_layer_call_fn_1124i:;Z[z{????<
5?2
,?)

input_data?????????  
p 
? "??????????
?
)__inference_cnn__model_layer_call_fn_1139f:;Z[z{??<?9
2?/
)?&
input_1?????????  
p 
? "??????????
?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_693142?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_693137?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_693120?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_693115?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_693098?:;Z[z{??A?>
? 
7?4
2
args_0(?%
args_0?????????  "3?0
.
output_1"?
output_1?????????
