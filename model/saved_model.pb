но#
╦»
B
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
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
delete_old_dirsbool(ѕ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ў
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12unknown8ўџ
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: @*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@ђ*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d└*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	d└*
dtype0
І
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└**
shared_namebatch_normalization/gamma
ё
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:└*
dtype0
Ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*)
shared_namebatch_normalization/beta
ѓ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:└*
dtype0
Ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*0
shared_name!batch_normalization/moving_mean
љ
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:└*
dtype0
Ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*4
shared_name%#batch_normalization/moving_variance
ў
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:└*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
ђђ*
dtype0
І
conv_s_n2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameconv_s_n2d_5/kernel
ё
'conv_s_n2d_5/kernel/Read/ReadVariableOpReadVariableOpconv_s_n2d_5/kernel*'
_output_shapes
:ђ*
dtype0
{
conv_s_n2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameconv_s_n2d_5/bias
t
%conv_s_n2d_5/bias/Read/ReadVariableOpReadVariableOpconv_s_n2d_5/bias*
_output_shapes	
:ђ*
dtype0
{
conv_s_n2d_5/snVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_nameconv_s_n2d_5/sn
t
#conv_s_n2d_5/sn/Read/ReadVariableOpReadVariableOpconv_s_n2d_5/sn*
_output_shapes
:	ђ*
dtype0
І
conv_s_n2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ@*$
shared_nameconv_s_n2d_6/kernel
ё
'conv_s_n2d_6/kernel/Read/ReadVariableOpReadVariableOpconv_s_n2d_6/kernel*'
_output_shapes
:ђ@*
dtype0
z
conv_s_n2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv_s_n2d_6/bias
s
%conv_s_n2d_6/bias/Read/ReadVariableOpReadVariableOpconv_s_n2d_6/bias*
_output_shapes
:@*
dtype0
z
conv_s_n2d_6/snVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_nameconv_s_n2d_6/sn
s
#conv_s_n2d_6/sn/Read/ReadVariableOpReadVariableOpconv_s_n2d_6/sn*
_output_shapes

:@*
dtype0
і
conv_s_n2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *$
shared_nameconv_s_n2d_7/kernel
Ѓ
'conv_s_n2d_7/kernel/Read/ReadVariableOpReadVariableOpconv_s_n2d_7/kernel*&
_output_shapes
:@ *
dtype0
z
conv_s_n2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv_s_n2d_7/bias
s
%conv_s_n2d_7/bias/Read/ReadVariableOpReadVariableOpconv_s_n2d_7/bias*
_output_shapes
: *
dtype0
z
conv_s_n2d_7/snVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_nameconv_s_n2d_7/sn
s
#conv_s_n2d_7/sn/Read/ReadVariableOpReadVariableOpconv_s_n2d_7/sn*
_output_shapes

: *
dtype0
і
conv_s_n2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv_s_n2d_8/kernel
Ѓ
'conv_s_n2d_8/kernel/Read/ReadVariableOpReadVariableOpconv_s_n2d_8/kernel*&
_output_shapes
: *
dtype0
z
conv_s_n2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv_s_n2d_8/bias
s
%conv_s_n2d_8/bias/Read/ReadVariableOpReadVariableOpconv_s_n2d_8/bias*
_output_shapes
:*
dtype0
z
conv_s_n2d_8/snVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameconv_s_n2d_8/sn
s
#conv_s_n2d_8/sn/Read/ReadVariableOpReadVariableOpconv_s_n2d_8/sn*
_output_shapes

:*
dtype0
і
conv_s_n2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv_s_n2d_9/kernel
Ѓ
'conv_s_n2d_9/kernel/Read/ReadVariableOpReadVariableOpconv_s_n2d_9/kernel*&
_output_shapes
:*
dtype0
z
conv_s_n2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv_s_n2d_9/bias
s
%conv_s_n2d_9/bias/Read/ReadVariableOpReadVariableOpconv_s_n2d_9/bias*
_output_shapes
:*
dtype0
z
conv_s_n2d_9/snVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nameconv_s_n2d_9/sn
s
#conv_s_n2d_9/sn/Read/ReadVariableOpReadVariableOpconv_s_n2d_9/sn*
_output_shapes

:*
dtype0

NoOpNoOp
эd
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*▓d
valueеdBЦd Bъd
Е
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer-21
layer_with_weights-8
layer-22
layer-23
layer-24
layer_with_weights-9
layer-25
layer-26
layer-27
layer_with_weights-10
layer-28
layer-29
layer-30
 trainable_variables
!	variables
"regularization_losses
#	keras_api
$
signatures
 
^

%kernel
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
^

.kernel
/trainable_variables
0	variables
1regularization_losses
2	keras_api
R
3trainable_variables
4	variables
5regularization_losses
6	keras_api
 
^

7kernel
8trainable_variables
9	variables
:regularization_losses
;	keras_api
^

<kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api
R
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
Ќ
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
^

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
R
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
R
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
R
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
w

gkernel
hbias
isn
iu
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
R
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
w

vkernel
wbias
xsn
xu
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
S
}trainable_variables
~	variables
regularization_losses
ђ	keras_api
V
Ђtrainable_variables
ѓ	variables
Ѓregularization_losses
ё	keras_api

Ёkernel
	єbias
Єsn
Єu
ѕtrainable_variables
Ѕ	variables
іregularization_losses
І	keras_api
V
їtrainable_variables
Ї	variables
јregularization_losses
Ј	keras_api
V
љtrainable_variables
Љ	variables
њregularization_losses
Њ	keras_api

ћkernel
	Ћbias
ќsn
ќu
Ќtrainable_variables
ў	variables
Ўregularization_losses
џ	keras_api
V
Џtrainable_variables
ю	variables
Юregularization_losses
ъ	keras_api
V
Ъtrainable_variables
а	variables
Аregularization_losses
б	keras_api

Бkernel
	цbias
Цsn
Цu
дtrainable_variables
Д	variables
еregularization_losses
Е	keras_api
V
фtrainable_variables
Ф	variables
гregularization_losses
Г	keras_api
V
«trainable_variables
»	variables
░regularization_losses
▒	keras_api
ё
%0
.1
72
<3
F4
G5
N6
g7
h8
v9
w10
Ё11
є12
ћ13
Ћ14
Б15
ц16
┐
%0
.1
72
<3
F4
G5
H6
I7
N8
g9
h10
i11
v12
w13
x14
Ё15
є16
Є17
ћ18
Ћ19
ќ20
Б21
ц22
Ц23
 
▓
 trainable_variables
!	variables
▓layers
│metrics
┤layer_metrics
 хlayer_regularization_losses
"regularization_losses
Хnon_trainable_variables
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

%0

%0
 
▓
&trainable_variables
'	variables
иlayers
Иmetrics
╣layer_metrics
 ║layer_regularization_losses
(regularization_losses
╗non_trainable_variables
 
 
 
▓
*trainable_variables
+	variables
╝layers
йmetrics
Йlayer_metrics
 ┐layer_regularization_losses
,regularization_losses
└non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

.0

.0
 
▓
/trainable_variables
0	variables
┴layers
┬metrics
├layer_metrics
 ─layer_regularization_losses
1regularization_losses
┼non_trainable_variables
 
 
 
▓
3trainable_variables
4	variables
кlayers
Кmetrics
╚layer_metrics
 ╔layer_regularization_losses
5regularization_losses
╩non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

70

70
 
▓
8trainable_variables
9	variables
╦layers
╠metrics
═layer_metrics
 ╬layer_regularization_losses
:regularization_losses
¤non_trainable_variables
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

<0

<0
 
▓
=trainable_variables
>	variables
лlayers
Лmetrics
мlayer_metrics
 Мlayer_regularization_losses
?regularization_losses
нnon_trainable_variables
 
 
 
▓
Atrainable_variables
B	variables
Нlayers
оmetrics
Оlayer_metrics
 пlayer_regularization_losses
Cregularization_losses
┘non_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
H2
I3
 
▓
Jtrainable_variables
K	variables
┌layers
█metrics
▄layer_metrics
 Пlayer_regularization_losses
Lregularization_losses
яnon_trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE

N0

N0
 
▓
Otrainable_variables
P	variables
▀layers
Яmetrics
рlayer_metrics
 Рlayer_regularization_losses
Qregularization_losses
сnon_trainable_variables
 
 
 
▓
Strainable_variables
T	variables
Сlayers
тmetrics
Тlayer_metrics
 уlayer_regularization_losses
Uregularization_losses
Уnon_trainable_variables
 
 
 
▓
Wtrainable_variables
X	variables
жlayers
Жmetrics
вlayer_metrics
 Вlayer_regularization_losses
Yregularization_losses
ьnon_trainable_variables
 
 
 
▓
[trainable_variables
\	variables
Ьlayers
№metrics
­layer_metrics
 ыlayer_regularization_losses
]regularization_losses
Ыnon_trainable_variables
 
 
 
▓
_trainable_variables
`	variables
зlayers
Зmetrics
шlayer_metrics
 Шlayer_regularization_losses
aregularization_losses
эnon_trainable_variables
 
 
 
▓
ctrainable_variables
d	variables
Эlayers
щmetrics
Щlayer_metrics
 чlayer_regularization_losses
eregularization_losses
Чnon_trainable_variables
_]
VARIABLE_VALUEconv_s_n2d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_s_n2d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_s_n2d_5/sn2layer_with_weights-6/sn/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
i2
 
▓
jtrainable_variables
k	variables
§layers
■metrics
 layer_metrics
 ђlayer_regularization_losses
lregularization_losses
Ђnon_trainable_variables
 
 
 
▓
ntrainable_variables
o	variables
ѓlayers
Ѓmetrics
ёlayer_metrics
 Ёlayer_regularization_losses
pregularization_losses
єnon_trainable_variables
 
 
 
▓
rtrainable_variables
s	variables
Єlayers
ѕmetrics
Ѕlayer_metrics
 іlayer_regularization_losses
tregularization_losses
Іnon_trainable_variables
_]
VARIABLE_VALUEconv_s_n2d_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_s_n2d_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_s_n2d_6/sn2layer_with_weights-7/sn/.ATTRIBUTES/VARIABLE_VALUE

v0
w1

v0
w1
x2
 
▓
ytrainable_variables
z	variables
їlayers
Їmetrics
јlayer_metrics
 Јlayer_regularization_losses
{regularization_losses
љnon_trainable_variables
 
 
 
▓
}trainable_variables
~	variables
Љlayers
њmetrics
Њlayer_metrics
 ћlayer_regularization_losses
regularization_losses
Ћnon_trainable_variables
 
 
 
х
Ђtrainable_variables
ѓ	variables
ќlayers
Ќmetrics
ўlayer_metrics
 Ўlayer_regularization_losses
Ѓregularization_losses
џnon_trainable_variables
_]
VARIABLE_VALUEconv_s_n2d_7/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_s_n2d_7/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_s_n2d_7/sn2layer_with_weights-8/sn/.ATTRIBUTES/VARIABLE_VALUE

Ё0
є1

Ё0
є1
Є2
 
х
ѕtrainable_variables
Ѕ	variables
Џlayers
юmetrics
Юlayer_metrics
 ъlayer_regularization_losses
іregularization_losses
Ъnon_trainable_variables
 
 
 
х
їtrainable_variables
Ї	variables
аlayers
Аmetrics
бlayer_metrics
 Бlayer_regularization_losses
јregularization_losses
цnon_trainable_variables
 
 
 
х
љtrainable_variables
Љ	variables
Цlayers
дmetrics
Дlayer_metrics
 еlayer_regularization_losses
њregularization_losses
Еnon_trainable_variables
_]
VARIABLE_VALUEconv_s_n2d_8/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_s_n2d_8/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_s_n2d_8/sn2layer_with_weights-9/sn/.ATTRIBUTES/VARIABLE_VALUE

ћ0
Ћ1

ћ0
Ћ1
ќ2
 
х
Ќtrainable_variables
ў	variables
фlayers
Фmetrics
гlayer_metrics
 Гlayer_regularization_losses
Ўregularization_losses
«non_trainable_variables
 
 
 
х
Џtrainable_variables
ю	variables
»layers
░metrics
▒layer_metrics
 ▓layer_regularization_losses
Юregularization_losses
│non_trainable_variables
 
 
 
х
Ъtrainable_variables
а	variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
Аregularization_losses
Иnon_trainable_variables
`^
VARIABLE_VALUEconv_s_n2d_9/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEconv_s_n2d_9/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_s_n2d_9/sn3layer_with_weights-10/sn/.ATTRIBUTES/VARIABLE_VALUE

Б0
ц1

Б0
ц1
Ц2
 
х
дtrainable_variables
Д	variables
╣layers
║metrics
╗layer_metrics
 ╝layer_regularization_losses
еregularization_losses
йnon_trainable_variables
 
 
 
х
фtrainable_variables
Ф	variables
Йlayers
┐metrics
└layer_metrics
 ┴layer_regularization_losses
гregularization_losses
┬non_trainable_variables
 
 
 
х
«trainable_variables
»	variables
├layers
─metrics
┼layer_metrics
 кlayer_regularization_losses
░regularization_losses
Кnon_trainable_variables
Ь
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 
 
 
4
H0
I1
i2
x3
Є4
ќ5
Ц6
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
H0
I1
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
 
 
 
 
 
 
 
 

i0
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
 

x0
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
 

Є0
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
 

ќ0
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
 

Ц0
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
z
serving_default_input_3Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
z
serving_default_input_4Placeholder*'
_output_shapes
:         d*
dtype0*
shape:         d
З
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4dense_1/kerneldense_2/kerneldense_3/kerneldense_5/kerneldense_4/kernel#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv_s_n2d_5/kernelconv_s_n2d_5/snconv_s_n2d_5/biasconv_s_n2d_6/kernelconv_s_n2d_6/snconv_s_n2d_6/biasconv_s_n2d_7/kernelconv_s_n2d_7/snconv_s_n2d_7/biasconv_s_n2d_8/kernelconv_s_n2d_8/snconv_s_n2d_8/biasconv_s_n2d_9/kernelconv_s_n2d_9/snconv_s_n2d_9/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђЭ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *.
f)R'
%__inference_signature_wrapper_1474125
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ђ

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp'conv_s_n2d_5/kernel/Read/ReadVariableOp%conv_s_n2d_5/bias/Read/ReadVariableOp#conv_s_n2d_5/sn/Read/ReadVariableOp'conv_s_n2d_6/kernel/Read/ReadVariableOp%conv_s_n2d_6/bias/Read/ReadVariableOp#conv_s_n2d_6/sn/Read/ReadVariableOp'conv_s_n2d_7/kernel/Read/ReadVariableOp%conv_s_n2d_7/bias/Read/ReadVariableOp#conv_s_n2d_7/sn/Read/ReadVariableOp'conv_s_n2d_8/kernel/Read/ReadVariableOp%conv_s_n2d_8/bias/Read/ReadVariableOp#conv_s_n2d_8/sn/Read/ReadVariableOp'conv_s_n2d_9/kernel/Read/ReadVariableOp%conv_s_n2d_9/bias/Read/ReadVariableOp#conv_s_n2d_9/sn/Read/ReadVariableOpConst*%
Tin
2*
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
GPU2*0,1J 8ѓ *)
f$R"
 __inference__traced_save_1475902
ю
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_2/kerneldense_3/kerneldense_5/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_4/kernelconv_s_n2d_5/kernelconv_s_n2d_5/biasconv_s_n2d_5/snconv_s_n2d_6/kernelconv_s_n2d_6/biasconv_s_n2d_6/snconv_s_n2d_7/kernelconv_s_n2d_7/biasconv_s_n2d_7/snconv_s_n2d_8/kernelconv_s_n2d_8/biasconv_s_n2d_8/snconv_s_n2d_9/kernelconv_s_n2d_9/biasconv_s_n2d_9/sn*$
Tin
2*
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
GPU2*0,1J 8ѓ *,
f'R%
#__inference__traced_restore_1475984ўв
Г
G
+__inference_reshape_1_layer_call_fn_1475095

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_14733882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▒
g
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_1475329

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           ђ2
	LeakyReluє
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Е
L
0__inference_leaky_re_lu_13_layer_call_fn_1475057

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_14733312
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
њ
.__inference_conv_s_n2d_6_layer_call_fn_1475442

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724792
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ы$
а
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1473497

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1Б
convolutionConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpј
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2	
BiasAddи
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
╝
е
5__inference_batch_normalization_layer_call_fn_1475010

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720332
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Ц
L
0__inference_leaky_re_lu_11_layer_call_fn_1474903

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_14732132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Њz
 
D__inference_model_1_layer_call_and_return_conditional_losses_1474018

inputs
inputs_1
dense_1_1473940
dense_2_1473944
dense_3_1473948
dense_5_1473952
dense_4_1473955
batch_normalization_1473958
batch_normalization_1473960
batch_normalization_1473962
batch_normalization_1473964
conv_s_n2d_5_1473972
conv_s_n2d_5_1473974
conv_s_n2d_5_1473976
conv_s_n2d_6_1473981
conv_s_n2d_6_1473983
conv_s_n2d_6_1473985
conv_s_n2d_7_1473990
conv_s_n2d_7_1473992
conv_s_n2d_7_1473994
conv_s_n2d_8_1473999
conv_s_n2d_8_1474001
conv_s_n2d_8_1474003
conv_s_n2d_9_1474008
conv_s_n2d_9_1474010
conv_s_n2d_9_1474012
identityѕб+batch_normalization/StatefulPartitionedCallб$conv_s_n2d_5/StatefulPartitionedCallб$conv_s_n2d_6/StatefulPartitionedCallб$conv_s_n2d_7/StatefulPartitionedCallб$conv_s_n2d_8/StatefulPartitionedCallб$conv_s_n2d_9/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallё
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_1473940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14731642!
dense_1/StatefulPartitionedCallј
leaky_re_lu_10/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_14731812 
leaky_re_lu_10/PartitionedCallЦ
dense_2/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0dense_2_1473944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_14731962!
dense_2/StatefulPartitionedCallј
leaky_re_lu_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_14732132 
leaky_re_lu_11/PartitionedCallд
dense_3/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0dense_3_1473948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_14732282!
dense_3/StatefulPartitionedCallЈ
leaky_re_lu_12/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_14732452 
leaky_re_lu_12/PartitionedCallЄ
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_5_1473952*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_14732602!
dense_5/StatefulPartitionedCallд
dense_4/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0dense_4_1473955*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_14732792!
dense_4/StatefulPartitionedCall┤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_1473958batch_normalization_1473960batch_normalization_1473962batch_normalization_1473964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720662-
+batch_normalization/StatefulPartitionedCallЈ
leaky_re_lu_13/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_14733312 
leaky_re_lu_13/PartitionedCallЏ
leaky_re_lu_14/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_14733442 
leaky_re_lu_14/PartitionedCallє
reshape_2/PartitionedCallPartitionedCall'leaky_re_lu_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_14733662
reshape_2/PartitionedCallє
reshape_1/PartitionedCallPartitionedCall'leaky_re_lu_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_14733882
reshape_1/PartitionedCall▓
concatenate_2/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_14734032
concatenate_2/PartitionedCallы
$conv_s_n2d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_s_n2d_5_1473972conv_s_n2d_5_1473974conv_s_n2d_5_1473976*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734972&
$conv_s_n2d_5/StatefulPartitionedCallФ
up_sampling2d/PartitionedCallPartitionedCall-conv_s_n2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_14722862
up_sampling2d/PartitionedCallД
leaky_re_lu_15/PartitionedCallPartitionedCall&up_sampling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_14735332 
leaky_re_lu_15/PartitionedCallЃ
$conv_s_n2d_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_15/PartitionedCall:output:0conv_s_n2d_6_1473981conv_s_n2d_6_1473983conv_s_n2d_6_1473985*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724792&
$conv_s_n2d_6/StatefulPartitionedCall░
up_sampling2d_1/PartitionedCallPartitionedCall-conv_s_n2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_14725012!
up_sampling2d_1/PartitionedCallе
leaky_re_lu_16/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_14735762 
leaky_re_lu_16/PartitionedCallЃ
$conv_s_n2d_7/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0conv_s_n2d_7_1473990conv_s_n2d_7_1473992conv_s_n2d_7_1473994*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726942&
$conv_s_n2d_7/StatefulPartitionedCall░
up_sampling2d_2/PartitionedCallPartitionedCall-conv_s_n2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_14727162!
up_sampling2d_2/PartitionedCallе
leaky_re_lu_17/PartitionedCallPartitionedCall(up_sampling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_14736192 
leaky_re_lu_17/PartitionedCallЃ
$conv_s_n2d_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv_s_n2d_8_1473999conv_s_n2d_8_1474001conv_s_n2d_8_1474003*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14729092&
$conv_s_n2d_8/StatefulPartitionedCall░
up_sampling2d_3/PartitionedCallPartitionedCall-conv_s_n2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14729312!
up_sampling2d_3/PartitionedCallе
leaky_re_lu_18/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_14736622 
leaky_re_lu_18/PartitionedCallЃ
$conv_s_n2d_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0conv_s_n2d_9_1474008conv_s_n2d_9_1474010conv_s_n2d_9_1474012*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14731242&
$conv_s_n2d_9/StatefulPartitionedCall░
up_sampling2d_4/PartitionedCallPartitionedCall-conv_s_n2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_14731462!
up_sampling2d_4/PartitionedCallе
leaky_re_lu_19/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_14737052 
leaky_re_lu_19/PartitionedCall░
IdentityIdentity'leaky_re_lu_19/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^conv_s_n2d_5/StatefulPartitionedCall%^conv_s_n2d_6/StatefulPartitionedCall%^conv_s_n2d_7/StatefulPartitionedCall%^conv_s_n2d_8/StatefulPartitionedCall%^conv_s_n2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$conv_s_n2d_5/StatefulPartitionedCall$conv_s_n2d_5/StatefulPartitionedCall2L
$conv_s_n2d_6/StatefulPartitionedCall$conv_s_n2d_6/StatefulPartitionedCall2L
$conv_s_n2d_7/StatefulPartitionedCall$conv_s_n2d_7/StatefulPartitionedCall2L
$conv_s_n2d_8/StatefulPartitionedCall$conv_s_n2d_8/StatefulPartitionedCall2L
$conv_s_n2d_9/StatefulPartitionedCall$conv_s_n2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_nameinputs
Й
е
5__inference_batch_normalization_layer_call_fn_1475023

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720662
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_1475683

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ы$
а
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475194

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1Б
convolutionConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpј
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2	
BiasAddи
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
э
Ъ
D__inference_dense_4_layer_call_and_return_conditional_losses_1475030

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
л'
│
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1472210

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1х
convolutionConv2DinputsReshape_1:output:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpа
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAdd▄
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ј
L
0__inference_leaky_re_lu_19_layer_call_fn_1475806

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_14737052
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ж
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1475071

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:          >2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
в
с
)__inference_model_1_layer_call_fn_1473933
input_3
input_4
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_14738822
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
К'
│
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475731

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	љ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	љ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	љ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	љ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	љ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	љ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц
L
0__inference_leaky_re_lu_10_layer_call_fn_1474879

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_14731812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ш
њ
.__inference_conv_s_n2d_5_layer_call_fn_1475313

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14722102
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ѕ
h
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1472931

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╚
g
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_1474936

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђ2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
З
њ
.__inference_conv_s_n2d_7_layer_call_fn_1475560

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726942
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
З
Ъ
D__inference_dense_3_layer_call_and_return_conditional_losses_1473228

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0**
_input_shapes
:         @:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ш
њ
.__inference_conv_s_n2d_6_layer_call_fn_1475431

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724252
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
з
њ
.__inference_conv_s_n2d_9_layer_call_fn_1475785

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14730702
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─
g
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_1473181

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:          2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┴
o
)__inference_dense_2_layer_call_fn_1474893

inputs
unknown
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_14731962
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0**
_input_shapes
:          :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬%
а
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1473124

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	љ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	љ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	љ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	љ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	љ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	љ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ј
L
0__inference_leaky_re_lu_18_layer_call_fn_1475688

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_14736622
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
п
[
/__inference_concatenate_2_layer_call_fn_1475108
inputs_0
inputs_1
identityР
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_14734032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          >:          >:Y U
/
_output_shapes
:          >
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          >
"
_user_specified_name
inputs/1
▒
g
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_1473533

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           ђ2
	LeakyReluє
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
З
Ъ
D__inference_dense_3_layer_call_and_return_conditional_losses_1474910

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0**
_input_shapes
:         @:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╚
g
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_1473331

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђ2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
№
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1475102
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЅ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:          >2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          >:          >:Y U
/
_output_shapes
:          >
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:          >
"
_user_specified_name
inputs/1
┤
M
1__inference_up_sampling2d_3_layer_call_fn_1472937

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14729312
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Э
т
)__inference_model_1_layer_call_fn_1474855
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_14740182
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs/1
у
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1473403

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЄ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:          >2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:          >:          >:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs:WS
/
_output_shapes
:          >
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_1473619

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
─
g
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_1474898

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         @2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
З
Ъ
D__inference_dense_5_layer_call_and_return_conditional_losses_1473260

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d└*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
р
Є
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1472066

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╚
g
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_1473245

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђ2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
к%
а
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1472479

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	ђ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	ђ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	ђ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:@2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:@2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:@2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Г
њ
.__inference_conv_s_n2d_5_layer_call_fn_1475205

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734542
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
┬%
а
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475656

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	а2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	а2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	а2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	а2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	а2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	а2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
K
/__inference_up_sampling2d_layer_call_fn_1472292

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_14722862
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г0
╔
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474977

inputs
assignmovingavg_1474952
assignmovingavg_1_1474958)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	└2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         └2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1═
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/1474952*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
AssignMovingAvg/decayЋ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1474952*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOpз
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/1474952*
_output_shapes	
:└2
AssignMovingAvg/subЖ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/1474952*
_output_shapes	
:└2
AssignMovingAvg/mul▒
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1474952AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/1474952*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpМ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/1474958*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
AssignMovingAvg_1/decayЏ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1474958*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp§
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1474958*
_output_shapes	
:└2
AssignMovingAvg_1/subЗ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1474958*
_output_shapes	
:└2
AssignMovingAvg_1/mulй
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1474958AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/1474958*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
┤
M
1__inference_up_sampling2d_4_layer_call_fn_1473152

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_14731462
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦%
а
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475302

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1х
convolutionConv2DinputsReshape_1:output:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpа
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAdd╔
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ј
L
0__inference_leaky_re_lu_17_layer_call_fn_1475570

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_14736192
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
─
g
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_1474874

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:          2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ы
Ъ
D__inference_dense_1_layer_call_and_return_conditional_losses_1473164

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:         :2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┤└
А
"__inference__wrapped_model_1471937
input_3
input_42
.model_1_dense_1_matmul_readvariableop_resource2
.model_1_dense_2_matmul_readvariableop_resource2
.model_1_dense_3_matmul_readvariableop_resource2
.model_1_dense_5_matmul_readvariableop_resource2
.model_1_dense_4_matmul_readvariableop_resourceA
=model_1_batch_normalization_batchnorm_readvariableop_resourceE
Amodel_1_batch_normalization_batchnorm_mul_readvariableop_resourceC
?model_1_batch_normalization_batchnorm_readvariableop_1_resourceC
?model_1_batch_normalization_batchnorm_readvariableop_2_resource8
4model_1_conv_s_n2d_5_reshape_readvariableop_resource7
3model_1_conv_s_n2d_5_matmul_readvariableop_resource8
4model_1_conv_s_n2d_5_biasadd_readvariableop_resource8
4model_1_conv_s_n2d_6_reshape_readvariableop_resource7
3model_1_conv_s_n2d_6_matmul_readvariableop_resource8
4model_1_conv_s_n2d_6_biasadd_readvariableop_resource8
4model_1_conv_s_n2d_7_reshape_readvariableop_resource7
3model_1_conv_s_n2d_7_matmul_readvariableop_resource8
4model_1_conv_s_n2d_7_biasadd_readvariableop_resource8
4model_1_conv_s_n2d_8_reshape_readvariableop_resource7
3model_1_conv_s_n2d_8_matmul_readvariableop_resource8
4model_1_conv_s_n2d_8_biasadd_readvariableop_resource8
4model_1_conv_s_n2d_9_reshape_readvariableop_resource7
3model_1_conv_s_n2d_9_matmul_readvariableop_resource8
4model_1_conv_s_n2d_9_biasadd_readvariableop_resource
identityѕб4model_1/batch_normalization/batchnorm/ReadVariableOpб6model_1/batch_normalization/batchnorm/ReadVariableOp_1б6model_1/batch_normalization/batchnorm/ReadVariableOp_2б8model_1/batch_normalization/batchnorm/mul/ReadVariableOpб+model_1/conv_s_n2d_5/BiasAdd/ReadVariableOpб*model_1/conv_s_n2d_5/MatMul/ReadVariableOpб+model_1/conv_s_n2d_5/Reshape/ReadVariableOpб+model_1/conv_s_n2d_6/BiasAdd/ReadVariableOpб*model_1/conv_s_n2d_6/MatMul/ReadVariableOpб+model_1/conv_s_n2d_6/Reshape/ReadVariableOpб+model_1/conv_s_n2d_7/BiasAdd/ReadVariableOpб*model_1/conv_s_n2d_7/MatMul/ReadVariableOpб+model_1/conv_s_n2d_7/Reshape/ReadVariableOpб+model_1/conv_s_n2d_8/BiasAdd/ReadVariableOpб*model_1/conv_s_n2d_8/MatMul/ReadVariableOpб+model_1/conv_s_n2d_8/Reshape/ReadVariableOpб+model_1/conv_s_n2d_9/BiasAdd/ReadVariableOpб*model_1/conv_s_n2d_9/MatMul/ReadVariableOpб+model_1/conv_s_n2d_9/Reshape/ReadVariableOpб%model_1/dense_1/MatMul/ReadVariableOpб%model_1/dense_2/MatMul/ReadVariableOpб%model_1/dense_3/MatMul/ReadVariableOpб%model_1/dense_4/MatMul/ReadVariableOpб%model_1/dense_5/MatMul/ReadVariableOpй
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_1/MatMul/ReadVariableOpц
model_1/dense_1/MatMulMatMulinput_3-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_1/MatMulю
 model_1/leaky_re_lu_10/LeakyRelu	LeakyRelu model_1/dense_1/MatMul:product:0*'
_output_shapes
:          2"
 model_1/leaky_re_lu_10/LeakyReluй
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02'
%model_1/dense_2/MatMul/ReadVariableOp╦
model_1/dense_2/MatMulMatMul.model_1/leaky_re_lu_10/LeakyRelu:activations:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model_1/dense_2/MatMulю
 model_1/leaky_re_lu_11/LeakyRelu	LeakyRelu model_1/dense_2/MatMul:product:0*'
_output_shapes
:         @2"
 model_1/leaky_re_lu_11/LeakyReluЙ
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp╠
model_1/dense_3/MatMulMatMul.model_1/leaky_re_lu_11/LeakyRelu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model_1/dense_3/MatMulЮ
 model_1/leaky_re_lu_12/LeakyRelu	LeakyRelu model_1/dense_3/MatMul:product:0*(
_output_shapes
:         ђ2"
 model_1/leaky_re_lu_12/LeakyReluЙ
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	d└*
dtype02'
%model_1/dense_5/MatMul/ReadVariableOpЦ
model_1/dense_5/MatMulMatMulinput_4-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
model_1/dense_5/MatMul┐
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp╠
model_1/dense_4/MatMulMatMul.model_1/leaky_re_lu_12/LeakyRelu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
model_1/dense_4/MatMulу
4model_1/batch_normalization/batchnorm/ReadVariableOpReadVariableOp=model_1_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype026
4model_1/batch_normalization/batchnorm/ReadVariableOpЪ
+model_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2-
+model_1/batch_normalization/batchnorm/add/yщ
)model_1/batch_normalization/batchnorm/addAddV2<model_1/batch_normalization/batchnorm/ReadVariableOp:value:04model_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2+
)model_1/batch_normalization/batchnorm/addИ
+model_1/batch_normalization/batchnorm/RsqrtRsqrt-model_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└2-
+model_1/batch_normalization/batchnorm/Rsqrtз
8model_1/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_1_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02:
8model_1/batch_normalization/batchnorm/mul/ReadVariableOpШ
)model_1/batch_normalization/batchnorm/mulMul/model_1/batch_normalization/batchnorm/Rsqrt:y:0@model_1/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2+
)model_1/batch_normalization/batchnorm/mulт
+model_1/batch_normalization/batchnorm/mul_1Mul model_1/dense_5/MatMul:product:0-model_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └2-
+model_1/batch_normalization/batchnorm/mul_1ь
6model_1/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp?model_1_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype028
6model_1/batch_normalization/batchnorm/ReadVariableOp_1Ш
+model_1/batch_normalization/batchnorm/mul_2Mul>model_1/batch_normalization/batchnorm/ReadVariableOp_1:value:0-model_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2-
+model_1/batch_normalization/batchnorm/mul_2ь
6model_1/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp?model_1_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype028
6model_1/batch_normalization/batchnorm/ReadVariableOp_2З
)model_1/batch_normalization/batchnorm/subSub>model_1/batch_normalization/batchnorm/ReadVariableOp_2:value:0/model_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2+
)model_1/batch_normalization/batchnorm/subШ
+model_1/batch_normalization/batchnorm/add_1AddV2/model_1/batch_normalization/batchnorm/mul_1:z:0-model_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2-
+model_1/batch_normalization/batchnorm/add_1Ю
 model_1/leaky_re_lu_13/LeakyRelu	LeakyRelu model_1/dense_4/MatMul:product:0*(
_output_shapes
:         ђ2"
 model_1/leaky_re_lu_13/LeakyReluг
 model_1/leaky_re_lu_14/LeakyRelu	LeakyRelu/model_1/batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         └2"
 model_1/leaky_re_lu_14/LeakyReluљ
model_1/reshape_2/ShapeShape.model_1/leaky_re_lu_14/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model_1/reshape_2/Shapeў
%model_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_2/strided_slice/stackю
'model_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_2/strided_slice/stack_1ю
'model_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_2/strided_slice/stack_2╬
model_1/reshape_2/strided_sliceStridedSlice model_1/reshape_2/Shape:output:0.model_1/reshape_2/strided_slice/stack:output:00model_1/reshape_2/strided_slice/stack_1:output:00model_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_2/strided_sliceѕ
!model_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!model_1/reshape_2/Reshape/shape/1ѕ
!model_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2#
!model_1/reshape_2/Reshape/shape/2ѕ
!model_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_2/Reshape/shape/3д
model_1/reshape_2/Reshape/shapePack(model_1/reshape_2/strided_slice:output:0*model_1/reshape_2/Reshape/shape/1:output:0*model_1/reshape_2/Reshape/shape/2:output:0*model_1/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_2/Reshape/shapeН
model_1/reshape_2/ReshapeReshape.model_1/leaky_re_lu_14/LeakyRelu:activations:0(model_1/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
model_1/reshape_2/Reshapeљ
model_1/reshape_1/ShapeShape.model_1/leaky_re_lu_13/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model_1/reshape_1/Shapeў
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_1/strided_slice/stackю
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_1ю
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_2╬
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_1/strided_sliceѕ
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!model_1/reshape_1/Reshape/shape/1ѕ
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2#
!model_1/reshape_1/Reshape/shape/2ѕ
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_1/Reshape/shape/3д
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_1/Reshape/shapeН
model_1/reshape_1/ReshapeReshape.model_1/leaky_re_lu_13/LeakyRelu:activations:0(model_1/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
model_1/reshape_1/Reshapeѕ
!model_1/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_2/concat/axis 
model_1/concatenate_2/concatConcatV2"model_1/reshape_2/Reshape:output:0"model_1/reshape_1/Reshape:output:0*model_1/concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:          >2
model_1/concatenate_2/concatп
+model_1/conv_s_n2d_5/Reshape/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_5_reshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02-
+model_1/conv_s_n2d_5/Reshape/ReadVariableOpЎ
"model_1/conv_s_n2d_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2$
"model_1/conv_s_n2d_5/Reshape/shapeМ
model_1/conv_s_n2d_5/ReshapeReshape3model_1/conv_s_n2d_5/Reshape/ReadVariableOp:value:0+model_1/conv_s_n2d_5/Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2
model_1/conv_s_n2d_5/ReshapeЏ
#model_1/conv_s_n2d_5/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_1/conv_s_n2d_5/transpose/perm╠
model_1/conv_s_n2d_5/transpose	Transpose%model_1/conv_s_n2d_5/Reshape:output:0,model_1/conv_s_n2d_5/transpose/perm:output:0*
T0*
_output_shapes
:	ђK2 
model_1/conv_s_n2d_5/transpose═
*model_1/conv_s_n2d_5/MatMul/ReadVariableOpReadVariableOp3model_1_conv_s_n2d_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02,
*model_1/conv_s_n2d_5/MatMul/ReadVariableOp┼
model_1/conv_s_n2d_5/MatMulMatMul2model_1/conv_s_n2d_5/MatMul/ReadVariableOp:value:0"model_1/conv_s_n2d_5/transpose:y:0*
T0*
_output_shapes

:K2
model_1/conv_s_n2d_5/MatMul}
model_1/conv_s_n2d_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_5/pow/y░
model_1/conv_s_n2d_5/powPow%model_1/conv_s_n2d_5/MatMul:product:0#model_1/conv_s_n2d_5/pow/y:output:0*
T0*
_output_shapes

:K2
model_1/conv_s_n2d_5/powЅ
model_1/conv_s_n2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_5/ConstЪ
model_1/conv_s_n2d_5/SumSummodel_1/conv_s_n2d_5/pow:z:0#model_1/conv_s_n2d_5/Const:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/SumЂ
model_1/conv_s_n2d_5/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_5/pow_1/yф
model_1/conv_s_n2d_5/pow_1Pow!model_1/conv_s_n2d_5/Sum:output:0%model_1/conv_s_n2d_5/pow_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/pow_1}
model_1/conv_s_n2d_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_5/add/yБ
model_1/conv_s_n2d_5/addAddV2model_1/conv_s_n2d_5/pow_1:z:0#model_1/conv_s_n2d_5/add/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/addх
model_1/conv_s_n2d_5/truedivRealDiv%model_1/conv_s_n2d_5/MatMul:product:0model_1/conv_s_n2d_5/add:z:0*
T0*
_output_shapes

:K2
model_1/conv_s_n2d_5/truediv╗
model_1/conv_s_n2d_5/MatMul_1MatMul model_1/conv_s_n2d_5/truediv:z:0%model_1/conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_5/MatMul_1Ђ
model_1/conv_s_n2d_5/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_5/pow_2/y╣
model_1/conv_s_n2d_5/pow_2Pow'model_1/conv_s_n2d_5/MatMul_1:product:0%model_1/conv_s_n2d_5/pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_5/pow_2Ї
model_1/conv_s_n2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_5/Const_1Д
model_1/conv_s_n2d_5/Sum_1Summodel_1/conv_s_n2d_5/pow_2:z:0%model_1/conv_s_n2d_5/Const_1:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/Sum_1Ђ
model_1/conv_s_n2d_5/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_5/pow_3/yг
model_1/conv_s_n2d_5/pow_3Pow#model_1/conv_s_n2d_5/Sum_1:output:0%model_1/conv_s_n2d_5/pow_3/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/pow_3Ђ
model_1/conv_s_n2d_5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_5/add_1/yЕ
model_1/conv_s_n2d_5/add_1AddV2model_1/conv_s_n2d_5/pow_3:z:0%model_1/conv_s_n2d_5/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_5/add_1Й
model_1/conv_s_n2d_5/truediv_1RealDiv'model_1/conv_s_n2d_5/MatMul_1:product:0model_1/conv_s_n2d_5/add_1:z:0*
T0*
_output_shapes
:	ђ2 
model_1/conv_s_n2d_5/truediv_1╗
model_1/conv_s_n2d_5/MatMul_2MatMul model_1/conv_s_n2d_5/truediv:z:0%model_1/conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_5/MatMul_2Ъ
%model_1/conv_s_n2d_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/conv_s_n2d_5/transpose_1/perm¤
 model_1/conv_s_n2d_5/transpose_1	Transpose"model_1/conv_s_n2d_5/truediv_1:z:0.model_1/conv_s_n2d_5/transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2"
 model_1/conv_s_n2d_5/transpose_1└
model_1/conv_s_n2d_5/MatMul_3MatMul'model_1/conv_s_n2d_5/MatMul_2:product:0$model_1/conv_s_n2d_5/transpose_1:y:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_5/MatMul_3┼
model_1/conv_s_n2d_5/truediv_2RealDiv%model_1/conv_s_n2d_5/Reshape:output:0'model_1/conv_s_n2d_5/MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2 
model_1/conv_s_n2d_5/truediv_2Ц
$model_1/conv_s_n2d_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2&
$model_1/conv_s_n2d_5/Reshape_1/shapeл
model_1/conv_s_n2d_5/Reshape_1Reshape"model_1/conv_s_n2d_5/truediv_2:z:0-model_1/conv_s_n2d_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2 
model_1/conv_s_n2d_5/Reshape_1Ђ
 model_1/conv_s_n2d_5/convolutionConv2D%model_1/concatenate_2/concat:output:0'model_1/conv_s_n2d_5/Reshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2"
 model_1/conv_s_n2d_5/convolution╠
+model_1/conv_s_n2d_5/BiasAdd/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_5_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+model_1/conv_s_n2d_5/BiasAdd/ReadVariableOpР
model_1/conv_s_n2d_5/BiasAddBiasAdd)model_1/conv_s_n2d_5/convolution:output:03model_1/conv_s_n2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2
model_1/conv_s_n2d_5/BiasAddЈ
model_1/up_sampling2d/ShapeShape%model_1/conv_s_n2d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d/Shapeа
)model_1/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)model_1/up_sampling2d/strided_slice/stackц
+model_1/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d/strided_slice/stack_1ц
+model_1/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d/strided_slice/stack_2м
#model_1/up_sampling2d/strided_sliceStridedSlice$model_1/up_sampling2d/Shape:output:02model_1/up_sampling2d/strided_slice/stack:output:04model_1/up_sampling2d/strided_slice/stack_1:output:04model_1/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#model_1/up_sampling2d/strided_sliceІ
model_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d/ConstХ
model_1/up_sampling2d/mulMul,model_1/up_sampling2d/strided_slice:output:0$model_1/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d/mulю
2model_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%model_1/conv_s_n2d_5/BiasAdd:output:0model_1/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:          >ђ*
half_pixel_centers(24
2model_1/up_sampling2d/resize/ResizeNearestNeighbor╚
 model_1/leaky_re_lu_15/LeakyRelu	LeakyReluCmodel_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*0
_output_shapes
:          >ђ2"
 model_1/leaky_re_lu_15/LeakyReluп
+model_1/conv_s_n2d_6/Reshape/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_6_reshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02-
+model_1/conv_s_n2d_6/Reshape/ReadVariableOpЎ
"model_1/conv_s_n2d_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2$
"model_1/conv_s_n2d_6/Reshape/shapeМ
model_1/conv_s_n2d_6/ReshapeReshape3model_1/conv_s_n2d_6/Reshape/ReadVariableOp:value:0+model_1/conv_s_n2d_6/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2
model_1/conv_s_n2d_6/ReshapeЏ
#model_1/conv_s_n2d_6/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_1/conv_s_n2d_6/transpose/perm╠
model_1/conv_s_n2d_6/transpose	Transpose%model_1/conv_s_n2d_6/Reshape:output:0,model_1/conv_s_n2d_6/transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2 
model_1/conv_s_n2d_6/transpose╠
*model_1/conv_s_n2d_6/MatMul/ReadVariableOpReadVariableOp3model_1_conv_s_n2d_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model_1/conv_s_n2d_6/MatMul/ReadVariableOpк
model_1/conv_s_n2d_6/MatMulMatMul2model_1/conv_s_n2d_6/MatMul/ReadVariableOp:value:0"model_1/conv_s_n2d_6/transpose:y:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_6/MatMul}
model_1/conv_s_n2d_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_6/pow/y▒
model_1/conv_s_n2d_6/powPow%model_1/conv_s_n2d_6/MatMul:product:0#model_1/conv_s_n2d_6/pow/y:output:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_6/powЅ
model_1/conv_s_n2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_6/ConstЪ
model_1/conv_s_n2d_6/SumSummodel_1/conv_s_n2d_6/pow:z:0#model_1/conv_s_n2d_6/Const:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/SumЂ
model_1/conv_s_n2d_6/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_6/pow_1/yф
model_1/conv_s_n2d_6/pow_1Pow!model_1/conv_s_n2d_6/Sum:output:0%model_1/conv_s_n2d_6/pow_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/pow_1}
model_1/conv_s_n2d_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_6/add/yБ
model_1/conv_s_n2d_6/addAddV2model_1/conv_s_n2d_6/pow_1:z:0#model_1/conv_s_n2d_6/add/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/addХ
model_1/conv_s_n2d_6/truedivRealDiv%model_1/conv_s_n2d_6/MatMul:product:0model_1/conv_s_n2d_6/add:z:0*
T0*
_output_shapes
:	ђ2
model_1/conv_s_n2d_6/truediv║
model_1/conv_s_n2d_6/MatMul_1MatMul model_1/conv_s_n2d_6/truediv:z:0%model_1/conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
model_1/conv_s_n2d_6/MatMul_1Ђ
model_1/conv_s_n2d_6/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_6/pow_2/yИ
model_1/conv_s_n2d_6/pow_2Pow'model_1/conv_s_n2d_6/MatMul_1:product:0%model_1/conv_s_n2d_6/pow_2/y:output:0*
T0*
_output_shapes

:@2
model_1/conv_s_n2d_6/pow_2Ї
model_1/conv_s_n2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_6/Const_1Д
model_1/conv_s_n2d_6/Sum_1Summodel_1/conv_s_n2d_6/pow_2:z:0%model_1/conv_s_n2d_6/Const_1:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/Sum_1Ђ
model_1/conv_s_n2d_6/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_6/pow_3/yг
model_1/conv_s_n2d_6/pow_3Pow#model_1/conv_s_n2d_6/Sum_1:output:0%model_1/conv_s_n2d_6/pow_3/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/pow_3Ђ
model_1/conv_s_n2d_6/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_6/add_1/yЕ
model_1/conv_s_n2d_6/add_1AddV2model_1/conv_s_n2d_6/pow_3:z:0%model_1/conv_s_n2d_6/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_6/add_1й
model_1/conv_s_n2d_6/truediv_1RealDiv'model_1/conv_s_n2d_6/MatMul_1:product:0model_1/conv_s_n2d_6/add_1:z:0*
T0*
_output_shapes

:@2 
model_1/conv_s_n2d_6/truediv_1║
model_1/conv_s_n2d_6/MatMul_2MatMul model_1/conv_s_n2d_6/truediv:z:0%model_1/conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
model_1/conv_s_n2d_6/MatMul_2Ъ
%model_1/conv_s_n2d_6/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/conv_s_n2d_6/transpose_1/perm╬
 model_1/conv_s_n2d_6/transpose_1	Transpose"model_1/conv_s_n2d_6/truediv_1:z:0.model_1/conv_s_n2d_6/transpose_1/perm:output:0*
T0*
_output_shapes

:@2"
 model_1/conv_s_n2d_6/transpose_1└
model_1/conv_s_n2d_6/MatMul_3MatMul'model_1/conv_s_n2d_6/MatMul_2:product:0$model_1/conv_s_n2d_6/transpose_1:y:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_6/MatMul_3┼
model_1/conv_s_n2d_6/truediv_2RealDiv%model_1/conv_s_n2d_6/Reshape:output:0'model_1/conv_s_n2d_6/MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2 
model_1/conv_s_n2d_6/truediv_2Ц
$model_1/conv_s_n2d_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2&
$model_1/conv_s_n2d_6/Reshape_1/shapeл
model_1/conv_s_n2d_6/Reshape_1Reshape"model_1/conv_s_n2d_6/truediv_2:z:0-model_1/conv_s_n2d_6/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2 
model_1/conv_s_n2d_6/Reshape_1Ѕ
 model_1/conv_s_n2d_6/convolutionConv2D.model_1/leaky_re_lu_15/LeakyRelu:activations:0'model_1/conv_s_n2d_6/Reshape_1:output:0*
T0*/
_output_shapes
:          >@*
paddingSAME*
strides
2"
 model_1/conv_s_n2d_6/convolution╦
+model_1/conv_s_n2d_6/BiasAdd/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model_1/conv_s_n2d_6/BiasAdd/ReadVariableOpр
model_1/conv_s_n2d_6/BiasAddBiasAdd)model_1/conv_s_n2d_6/convolution:output:03model_1/conv_s_n2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          >@2
model_1/conv_s_n2d_6/BiasAddЊ
model_1/up_sampling2d_1/ShapeShape%model_1/conv_s_n2d_6/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_1/Shapeц
+model_1/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_1/strided_slice/stackе
-model_1/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_1/strided_slice/stack_1е
-model_1/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_1/strided_slice/stack_2я
%model_1/up_sampling2d_1/strided_sliceStridedSlice&model_1/up_sampling2d_1/Shape:output:04model_1/up_sampling2d_1/strided_slice/stack:output:06model_1/up_sampling2d_1/strided_slice/stack_1:output:06model_1/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_1/strided_sliceЈ
model_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_1/ConstЙ
model_1/up_sampling2d_1/mulMul.model_1/up_sampling2d_1/strided_slice:output:0&model_1/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_1/mulА
4model_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%model_1/conv_s_n2d_6/BiasAdd:output:0model_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:         @|@*
half_pixel_centers(26
4model_1/up_sampling2d_1/resize/ResizeNearestNeighbor╔
 model_1/leaky_re_lu_16/LeakyRelu	LeakyReluEmodel_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*/
_output_shapes
:         @|@2"
 model_1/leaky_re_lu_16/LeakyReluО
+model_1/conv_s_n2d_7/Reshape/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_7_reshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+model_1/conv_s_n2d_7/Reshape/ReadVariableOpЎ
"model_1/conv_s_n2d_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2$
"model_1/conv_s_n2d_7/Reshape/shapeМ
model_1/conv_s_n2d_7/ReshapeReshape3model_1/conv_s_n2d_7/Reshape/ReadVariableOp:value:0+model_1/conv_s_n2d_7/Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2
model_1/conv_s_n2d_7/ReshapeЏ
#model_1/conv_s_n2d_7/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_1/conv_s_n2d_7/transpose/perm╠
model_1/conv_s_n2d_7/transpose	Transpose%model_1/conv_s_n2d_7/Reshape:output:0,model_1/conv_s_n2d_7/transpose/perm:output:0*
T0*
_output_shapes
:	 └2 
model_1/conv_s_n2d_7/transpose╠
*model_1/conv_s_n2d_7/MatMul/ReadVariableOpReadVariableOp3model_1_conv_s_n2d_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*model_1/conv_s_n2d_7/MatMul/ReadVariableOpк
model_1/conv_s_n2d_7/MatMulMatMul2model_1/conv_s_n2d_7/MatMul/ReadVariableOp:value:0"model_1/conv_s_n2d_7/transpose:y:0*
T0*
_output_shapes
:	└2
model_1/conv_s_n2d_7/MatMul}
model_1/conv_s_n2d_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_7/pow/y▒
model_1/conv_s_n2d_7/powPow%model_1/conv_s_n2d_7/MatMul:product:0#model_1/conv_s_n2d_7/pow/y:output:0*
T0*
_output_shapes
:	└2
model_1/conv_s_n2d_7/powЅ
model_1/conv_s_n2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_7/ConstЪ
model_1/conv_s_n2d_7/SumSummodel_1/conv_s_n2d_7/pow:z:0#model_1/conv_s_n2d_7/Const:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/SumЂ
model_1/conv_s_n2d_7/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_7/pow_1/yф
model_1/conv_s_n2d_7/pow_1Pow!model_1/conv_s_n2d_7/Sum:output:0%model_1/conv_s_n2d_7/pow_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/pow_1}
model_1/conv_s_n2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_7/add/yБ
model_1/conv_s_n2d_7/addAddV2model_1/conv_s_n2d_7/pow_1:z:0#model_1/conv_s_n2d_7/add/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/addХ
model_1/conv_s_n2d_7/truedivRealDiv%model_1/conv_s_n2d_7/MatMul:product:0model_1/conv_s_n2d_7/add:z:0*
T0*
_output_shapes
:	└2
model_1/conv_s_n2d_7/truediv║
model_1/conv_s_n2d_7/MatMul_1MatMul model_1/conv_s_n2d_7/truediv:z:0%model_1/conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
model_1/conv_s_n2d_7/MatMul_1Ђ
model_1/conv_s_n2d_7/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_7/pow_2/yИ
model_1/conv_s_n2d_7/pow_2Pow'model_1/conv_s_n2d_7/MatMul_1:product:0%model_1/conv_s_n2d_7/pow_2/y:output:0*
T0*
_output_shapes

: 2
model_1/conv_s_n2d_7/pow_2Ї
model_1/conv_s_n2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_7/Const_1Д
model_1/conv_s_n2d_7/Sum_1Summodel_1/conv_s_n2d_7/pow_2:z:0%model_1/conv_s_n2d_7/Const_1:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/Sum_1Ђ
model_1/conv_s_n2d_7/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_7/pow_3/yг
model_1/conv_s_n2d_7/pow_3Pow#model_1/conv_s_n2d_7/Sum_1:output:0%model_1/conv_s_n2d_7/pow_3/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/pow_3Ђ
model_1/conv_s_n2d_7/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_7/add_1/yЕ
model_1/conv_s_n2d_7/add_1AddV2model_1/conv_s_n2d_7/pow_3:z:0%model_1/conv_s_n2d_7/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_7/add_1й
model_1/conv_s_n2d_7/truediv_1RealDiv'model_1/conv_s_n2d_7/MatMul_1:product:0model_1/conv_s_n2d_7/add_1:z:0*
T0*
_output_shapes

: 2 
model_1/conv_s_n2d_7/truediv_1║
model_1/conv_s_n2d_7/MatMul_2MatMul model_1/conv_s_n2d_7/truediv:z:0%model_1/conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
model_1/conv_s_n2d_7/MatMul_2Ъ
%model_1/conv_s_n2d_7/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/conv_s_n2d_7/transpose_1/perm╬
 model_1/conv_s_n2d_7/transpose_1	Transpose"model_1/conv_s_n2d_7/truediv_1:z:0.model_1/conv_s_n2d_7/transpose_1/perm:output:0*
T0*
_output_shapes

: 2"
 model_1/conv_s_n2d_7/transpose_1└
model_1/conv_s_n2d_7/MatMul_3MatMul'model_1/conv_s_n2d_7/MatMul_2:product:0$model_1/conv_s_n2d_7/transpose_1:y:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_7/MatMul_3┼
model_1/conv_s_n2d_7/truediv_2RealDiv%model_1/conv_s_n2d_7/Reshape:output:0'model_1/conv_s_n2d_7/MatMul_3:product:0*
T0*
_output_shapes
:	└ 2 
model_1/conv_s_n2d_7/truediv_2Ц
$model_1/conv_s_n2d_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       2&
$model_1/conv_s_n2d_7/Reshape_1/shape¤
model_1/conv_s_n2d_7/Reshape_1Reshape"model_1/conv_s_n2d_7/truediv_2:z:0-model_1/conv_s_n2d_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2 
model_1/conv_s_n2d_7/Reshape_1Ѕ
 model_1/conv_s_n2d_7/convolutionConv2D.model_1/leaky_re_lu_16/LeakyRelu:activations:0'model_1/conv_s_n2d_7/Reshape_1:output:0*
T0*/
_output_shapes
:         @| *
paddingSAME*
strides
2"
 model_1/conv_s_n2d_7/convolution╦
+model_1/conv_s_n2d_7/BiasAdd/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_1/conv_s_n2d_7/BiasAdd/ReadVariableOpр
model_1/conv_s_n2d_7/BiasAddBiasAdd)model_1/conv_s_n2d_7/convolution:output:03model_1/conv_s_n2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @| 2
model_1/conv_s_n2d_7/BiasAddЊ
model_1/up_sampling2d_2/ShapeShape%model_1/conv_s_n2d_7/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_2/Shapeц
+model_1/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_2/strided_slice/stackе
-model_1/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_2/strided_slice/stack_1е
-model_1/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_2/strided_slice/stack_2я
%model_1/up_sampling2d_2/strided_sliceStridedSlice&model_1/up_sampling2d_2/Shape:output:04model_1/up_sampling2d_2/strided_slice/stack:output:06model_1/up_sampling2d_2/strided_slice/stack_1:output:06model_1/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_2/strided_sliceЈ
model_1/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_2/ConstЙ
model_1/up_sampling2d_2/mulMul.model_1/up_sampling2d_2/strided_slice:output:0&model_1/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_2/mulБ
4model_1/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%model_1/conv_s_n2d_7/BiasAdd:output:0model_1/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђЭ *
half_pixel_centers(26
4model_1/up_sampling2d_2/resize/ResizeNearestNeighbor╦
 model_1/leaky_re_lu_17/LeakyRelu	LeakyReluEmodel_1/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ 2"
 model_1/leaky_re_lu_17/LeakyReluО
+model_1/conv_s_n2d_8/Reshape/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_8_reshape_readvariableop_resource*&
_output_shapes
: *
dtype02-
+model_1/conv_s_n2d_8/Reshape/ReadVariableOpЎ
"model_1/conv_s_n2d_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_1/conv_s_n2d_8/Reshape/shapeМ
model_1/conv_s_n2d_8/ReshapeReshape3model_1/conv_s_n2d_8/Reshape/ReadVariableOp:value:0+model_1/conv_s_n2d_8/Reshape/shape:output:0*
T0*
_output_shapes
:	а2
model_1/conv_s_n2d_8/ReshapeЏ
#model_1/conv_s_n2d_8/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_1/conv_s_n2d_8/transpose/perm╠
model_1/conv_s_n2d_8/transpose	Transpose%model_1/conv_s_n2d_8/Reshape:output:0,model_1/conv_s_n2d_8/transpose/perm:output:0*
T0*
_output_shapes
:	а2 
model_1/conv_s_n2d_8/transpose╠
*model_1/conv_s_n2d_8/MatMul/ReadVariableOpReadVariableOp3model_1_conv_s_n2d_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_1/conv_s_n2d_8/MatMul/ReadVariableOpк
model_1/conv_s_n2d_8/MatMulMatMul2model_1/conv_s_n2d_8/MatMul/ReadVariableOp:value:0"model_1/conv_s_n2d_8/transpose:y:0*
T0*
_output_shapes
:	а2
model_1/conv_s_n2d_8/MatMul}
model_1/conv_s_n2d_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_8/pow/y▒
model_1/conv_s_n2d_8/powPow%model_1/conv_s_n2d_8/MatMul:product:0#model_1/conv_s_n2d_8/pow/y:output:0*
T0*
_output_shapes
:	а2
model_1/conv_s_n2d_8/powЅ
model_1/conv_s_n2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_8/ConstЪ
model_1/conv_s_n2d_8/SumSummodel_1/conv_s_n2d_8/pow:z:0#model_1/conv_s_n2d_8/Const:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/SumЂ
model_1/conv_s_n2d_8/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_8/pow_1/yф
model_1/conv_s_n2d_8/pow_1Pow!model_1/conv_s_n2d_8/Sum:output:0%model_1/conv_s_n2d_8/pow_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/pow_1}
model_1/conv_s_n2d_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_8/add/yБ
model_1/conv_s_n2d_8/addAddV2model_1/conv_s_n2d_8/pow_1:z:0#model_1/conv_s_n2d_8/add/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/addХ
model_1/conv_s_n2d_8/truedivRealDiv%model_1/conv_s_n2d_8/MatMul:product:0model_1/conv_s_n2d_8/add:z:0*
T0*
_output_shapes
:	а2
model_1/conv_s_n2d_8/truediv║
model_1/conv_s_n2d_8/MatMul_1MatMul model_1/conv_s_n2d_8/truediv:z:0%model_1/conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_8/MatMul_1Ђ
model_1/conv_s_n2d_8/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_8/pow_2/yИ
model_1/conv_s_n2d_8/pow_2Pow'model_1/conv_s_n2d_8/MatMul_1:product:0%model_1/conv_s_n2d_8/pow_2/y:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_8/pow_2Ї
model_1/conv_s_n2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_8/Const_1Д
model_1/conv_s_n2d_8/Sum_1Summodel_1/conv_s_n2d_8/pow_2:z:0%model_1/conv_s_n2d_8/Const_1:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/Sum_1Ђ
model_1/conv_s_n2d_8/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_8/pow_3/yг
model_1/conv_s_n2d_8/pow_3Pow#model_1/conv_s_n2d_8/Sum_1:output:0%model_1/conv_s_n2d_8/pow_3/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/pow_3Ђ
model_1/conv_s_n2d_8/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_8/add_1/yЕ
model_1/conv_s_n2d_8/add_1AddV2model_1/conv_s_n2d_8/pow_3:z:0%model_1/conv_s_n2d_8/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_8/add_1й
model_1/conv_s_n2d_8/truediv_1RealDiv'model_1/conv_s_n2d_8/MatMul_1:product:0model_1/conv_s_n2d_8/add_1:z:0*
T0*
_output_shapes

:2 
model_1/conv_s_n2d_8/truediv_1║
model_1/conv_s_n2d_8/MatMul_2MatMul model_1/conv_s_n2d_8/truediv:z:0%model_1/conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_8/MatMul_2Ъ
%model_1/conv_s_n2d_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/conv_s_n2d_8/transpose_1/perm╬
 model_1/conv_s_n2d_8/transpose_1	Transpose"model_1/conv_s_n2d_8/truediv_1:z:0.model_1/conv_s_n2d_8/transpose_1/perm:output:0*
T0*
_output_shapes

:2"
 model_1/conv_s_n2d_8/transpose_1└
model_1/conv_s_n2d_8/MatMul_3MatMul'model_1/conv_s_n2d_8/MatMul_2:product:0$model_1/conv_s_n2d_8/transpose_1:y:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_8/MatMul_3┼
model_1/conv_s_n2d_8/truediv_2RealDiv%model_1/conv_s_n2d_8/Reshape:output:0'model_1/conv_s_n2d_8/MatMul_3:product:0*
T0*
_output_shapes
:	а2 
model_1/conv_s_n2d_8/truediv_2Ц
$model_1/conv_s_n2d_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2&
$model_1/conv_s_n2d_8/Reshape_1/shape¤
model_1/conv_s_n2d_8/Reshape_1Reshape"model_1/conv_s_n2d_8/truediv_2:z:0-model_1/conv_s_n2d_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2 
model_1/conv_s_n2d_8/Reshape_1І
 model_1/conv_s_n2d_8/convolutionConv2D.model_1/leaky_re_lu_17/LeakyRelu:activations:0'model_1/conv_s_n2d_8/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2"
 model_1/conv_s_n2d_8/convolution╦
+model_1/conv_s_n2d_8/BiasAdd/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_1/conv_s_n2d_8/BiasAdd/ReadVariableOpс
model_1/conv_s_n2d_8/BiasAddBiasAdd)model_1/conv_s_n2d_8/convolution:output:03model_1/conv_s_n2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
model_1/conv_s_n2d_8/BiasAddЊ
model_1/up_sampling2d_3/ShapeShape%model_1/conv_s_n2d_8/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_3/Shapeц
+model_1/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_3/strided_slice/stackе
-model_1/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_3/strided_slice/stack_1е
-model_1/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_3/strided_slice/stack_2я
%model_1/up_sampling2d_3/strided_sliceStridedSlice&model_1/up_sampling2d_3/Shape:output:04model_1/up_sampling2d_3/strided_slice/stack:output:06model_1/up_sampling2d_3/strided_slice/stack_1:output:06model_1/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_3/strided_sliceЈ
model_1/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_3/ConstЙ
model_1/up_sampling2d_3/mulMul.model_1/up_sampling2d_3/strided_slice:output:0&model_1/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_3/mulБ
4model_1/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%model_1/conv_s_n2d_8/BiasAdd:output:0model_1/up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(26
4model_1/up_sampling2d_3/resize/ResizeNearestNeighbor╦
 model_1/leaky_re_lu_18/LeakyRelu	LeakyReluEmodel_1/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2"
 model_1/leaky_re_lu_18/LeakyReluО
+model_1/conv_s_n2d_9/Reshape/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_9_reshape_readvariableop_resource*&
_output_shapes
:*
dtype02-
+model_1/conv_s_n2d_9/Reshape/ReadVariableOpЎ
"model_1/conv_s_n2d_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_1/conv_s_n2d_9/Reshape/shapeМ
model_1/conv_s_n2d_9/ReshapeReshape3model_1/conv_s_n2d_9/Reshape/ReadVariableOp:value:0+model_1/conv_s_n2d_9/Reshape/shape:output:0*
T0*
_output_shapes
:	љ2
model_1/conv_s_n2d_9/ReshapeЏ
#model_1/conv_s_n2d_9/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_1/conv_s_n2d_9/transpose/perm╠
model_1/conv_s_n2d_9/transpose	Transpose%model_1/conv_s_n2d_9/Reshape:output:0,model_1/conv_s_n2d_9/transpose/perm:output:0*
T0*
_output_shapes
:	љ2 
model_1/conv_s_n2d_9/transpose╠
*model_1/conv_s_n2d_9/MatMul/ReadVariableOpReadVariableOp3model_1_conv_s_n2d_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_1/conv_s_n2d_9/MatMul/ReadVariableOpк
model_1/conv_s_n2d_9/MatMulMatMul2model_1/conv_s_n2d_9/MatMul/ReadVariableOp:value:0"model_1/conv_s_n2d_9/transpose:y:0*
T0*
_output_shapes
:	љ2
model_1/conv_s_n2d_9/MatMul}
model_1/conv_s_n2d_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_9/pow/y▒
model_1/conv_s_n2d_9/powPow%model_1/conv_s_n2d_9/MatMul:product:0#model_1/conv_s_n2d_9/pow/y:output:0*
T0*
_output_shapes
:	љ2
model_1/conv_s_n2d_9/powЅ
model_1/conv_s_n2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_9/ConstЪ
model_1/conv_s_n2d_9/SumSummodel_1/conv_s_n2d_9/pow:z:0#model_1/conv_s_n2d_9/Const:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/SumЂ
model_1/conv_s_n2d_9/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_9/pow_1/yф
model_1/conv_s_n2d_9/pow_1Pow!model_1/conv_s_n2d_9/Sum:output:0%model_1/conv_s_n2d_9/pow_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/pow_1}
model_1/conv_s_n2d_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_9/add/yБ
model_1/conv_s_n2d_9/addAddV2model_1/conv_s_n2d_9/pow_1:z:0#model_1/conv_s_n2d_9/add/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/addХ
model_1/conv_s_n2d_9/truedivRealDiv%model_1/conv_s_n2d_9/MatMul:product:0model_1/conv_s_n2d_9/add:z:0*
T0*
_output_shapes
:	љ2
model_1/conv_s_n2d_9/truediv║
model_1/conv_s_n2d_9/MatMul_1MatMul model_1/conv_s_n2d_9/truediv:z:0%model_1/conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_9/MatMul_1Ђ
model_1/conv_s_n2d_9/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_1/conv_s_n2d_9/pow_2/yИ
model_1/conv_s_n2d_9/pow_2Pow'model_1/conv_s_n2d_9/MatMul_1:product:0%model_1/conv_s_n2d_9/pow_2/y:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_9/pow_2Ї
model_1/conv_s_n2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
model_1/conv_s_n2d_9/Const_1Д
model_1/conv_s_n2d_9/Sum_1Summodel_1/conv_s_n2d_9/pow_2:z:0%model_1/conv_s_n2d_9/Const_1:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/Sum_1Ђ
model_1/conv_s_n2d_9/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_1/conv_s_n2d_9/pow_3/yг
model_1/conv_s_n2d_9/pow_3Pow#model_1/conv_s_n2d_9/Sum_1:output:0%model_1/conv_s_n2d_9/pow_3/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/pow_3Ђ
model_1/conv_s_n2d_9/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
model_1/conv_s_n2d_9/add_1/yЕ
model_1/conv_s_n2d_9/add_1AddV2model_1/conv_s_n2d_9/pow_3:z:0%model_1/conv_s_n2d_9/add_1/y:output:0*
T0*
_output_shapes
: 2
model_1/conv_s_n2d_9/add_1й
model_1/conv_s_n2d_9/truediv_1RealDiv'model_1/conv_s_n2d_9/MatMul_1:product:0model_1/conv_s_n2d_9/add_1:z:0*
T0*
_output_shapes

:2 
model_1/conv_s_n2d_9/truediv_1║
model_1/conv_s_n2d_9/MatMul_2MatMul model_1/conv_s_n2d_9/truediv:z:0%model_1/conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_9/MatMul_2Ъ
%model_1/conv_s_n2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/conv_s_n2d_9/transpose_1/perm╬
 model_1/conv_s_n2d_9/transpose_1	Transpose"model_1/conv_s_n2d_9/truediv_1:z:0.model_1/conv_s_n2d_9/transpose_1/perm:output:0*
T0*
_output_shapes

:2"
 model_1/conv_s_n2d_9/transpose_1└
model_1/conv_s_n2d_9/MatMul_3MatMul'model_1/conv_s_n2d_9/MatMul_2:product:0$model_1/conv_s_n2d_9/transpose_1:y:0*
T0*
_output_shapes

:2
model_1/conv_s_n2d_9/MatMul_3┼
model_1/conv_s_n2d_9/truediv_2RealDiv%model_1/conv_s_n2d_9/Reshape:output:0'model_1/conv_s_n2d_9/MatMul_3:product:0*
T0*
_output_shapes
:	љ2 
model_1/conv_s_n2d_9/truediv_2Ц
$model_1/conv_s_n2d_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2&
$model_1/conv_s_n2d_9/Reshape_1/shape¤
model_1/conv_s_n2d_9/Reshape_1Reshape"model_1/conv_s_n2d_9/truediv_2:z:0-model_1/conv_s_n2d_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2 
model_1/conv_s_n2d_9/Reshape_1І
 model_1/conv_s_n2d_9/convolutionConv2D.model_1/leaky_re_lu_18/LeakyRelu:activations:0'model_1/conv_s_n2d_9/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2"
 model_1/conv_s_n2d_9/convolution╦
+model_1/conv_s_n2d_9/BiasAdd/ReadVariableOpReadVariableOp4model_1_conv_s_n2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_1/conv_s_n2d_9/BiasAdd/ReadVariableOpс
model_1/conv_s_n2d_9/BiasAddBiasAdd)model_1/conv_s_n2d_9/convolution:output:03model_1/conv_s_n2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
model_1/conv_s_n2d_9/BiasAddЊ
model_1/up_sampling2d_4/ShapeShape%model_1/conv_s_n2d_9/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_4/Shapeц
+model_1/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_4/strided_slice/stackе
-model_1/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_4/strided_slice/stack_1е
-model_1/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_4/strided_slice/stack_2я
%model_1/up_sampling2d_4/strided_sliceStridedSlice&model_1/up_sampling2d_4/Shape:output:04model_1/up_sampling2d_4/strided_slice/stack:output:06model_1/up_sampling2d_4/strided_slice/stack_1:output:06model_1/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_4/strided_sliceЈ
model_1/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_4/ConstЙ
model_1/up_sampling2d_4/mulMul.model_1/up_sampling2d_4/strided_slice:output:0&model_1/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_4/mulБ
4model_1/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor%model_1/conv_s_n2d_9/BiasAdd:output:0model_1/up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(26
4model_1/up_sampling2d_4/resize/ResizeNearestNeighbor╦
 model_1/leaky_re_lu_19/LeakyRelu	LeakyReluEmodel_1/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2"
 model_1/leaky_re_lu_19/LeakyReluт	
IdentityIdentity.model_1/leaky_re_lu_19/LeakyRelu:activations:05^model_1/batch_normalization/batchnorm/ReadVariableOp7^model_1/batch_normalization/batchnorm/ReadVariableOp_17^model_1/batch_normalization/batchnorm/ReadVariableOp_29^model_1/batch_normalization/batchnorm/mul/ReadVariableOp,^model_1/conv_s_n2d_5/BiasAdd/ReadVariableOp+^model_1/conv_s_n2d_5/MatMul/ReadVariableOp,^model_1/conv_s_n2d_5/Reshape/ReadVariableOp,^model_1/conv_s_n2d_6/BiasAdd/ReadVariableOp+^model_1/conv_s_n2d_6/MatMul/ReadVariableOp,^model_1/conv_s_n2d_6/Reshape/ReadVariableOp,^model_1/conv_s_n2d_7/BiasAdd/ReadVariableOp+^model_1/conv_s_n2d_7/MatMul/ReadVariableOp,^model_1/conv_s_n2d_7/Reshape/ReadVariableOp,^model_1/conv_s_n2d_8/BiasAdd/ReadVariableOp+^model_1/conv_s_n2d_8/MatMul/ReadVariableOp,^model_1/conv_s_n2d_8/Reshape/ReadVariableOp,^model_1/conv_s_n2d_9/BiasAdd/ReadVariableOp+^model_1/conv_s_n2d_9/MatMul/ReadVariableOp,^model_1/conv_s_n2d_9/Reshape/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*
T0*1
_output_shapes
:         ђЭ2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2l
4model_1/batch_normalization/batchnorm/ReadVariableOp4model_1/batch_normalization/batchnorm/ReadVariableOp2p
6model_1/batch_normalization/batchnorm/ReadVariableOp_16model_1/batch_normalization/batchnorm/ReadVariableOp_12p
6model_1/batch_normalization/batchnorm/ReadVariableOp_26model_1/batch_normalization/batchnorm/ReadVariableOp_22t
8model_1/batch_normalization/batchnorm/mul/ReadVariableOp8model_1/batch_normalization/batchnorm/mul/ReadVariableOp2Z
+model_1/conv_s_n2d_5/BiasAdd/ReadVariableOp+model_1/conv_s_n2d_5/BiasAdd/ReadVariableOp2X
*model_1/conv_s_n2d_5/MatMul/ReadVariableOp*model_1/conv_s_n2d_5/MatMul/ReadVariableOp2Z
+model_1/conv_s_n2d_5/Reshape/ReadVariableOp+model_1/conv_s_n2d_5/Reshape/ReadVariableOp2Z
+model_1/conv_s_n2d_6/BiasAdd/ReadVariableOp+model_1/conv_s_n2d_6/BiasAdd/ReadVariableOp2X
*model_1/conv_s_n2d_6/MatMul/ReadVariableOp*model_1/conv_s_n2d_6/MatMul/ReadVariableOp2Z
+model_1/conv_s_n2d_6/Reshape/ReadVariableOp+model_1/conv_s_n2d_6/Reshape/ReadVariableOp2Z
+model_1/conv_s_n2d_7/BiasAdd/ReadVariableOp+model_1/conv_s_n2d_7/BiasAdd/ReadVariableOp2X
*model_1/conv_s_n2d_7/MatMul/ReadVariableOp*model_1/conv_s_n2d_7/MatMul/ReadVariableOp2Z
+model_1/conv_s_n2d_7/Reshape/ReadVariableOp+model_1/conv_s_n2d_7/Reshape/ReadVariableOp2Z
+model_1/conv_s_n2d_8/BiasAdd/ReadVariableOp+model_1/conv_s_n2d_8/BiasAdd/ReadVariableOp2X
*model_1/conv_s_n2d_8/MatMul/ReadVariableOp*model_1/conv_s_n2d_8/MatMul/ReadVariableOp2Z
+model_1/conv_s_n2d_8/Reshape/ReadVariableOp+model_1/conv_s_n2d_8/Reshape/ReadVariableOp2Z
+model_1/conv_s_n2d_9/BiasAdd/ReadVariableOp+model_1/conv_s_n2d_9/BiasAdd/ReadVariableOp2X
*model_1/conv_s_n2d_9/MatMul/ReadVariableOp*model_1/conv_s_n2d_9/MatMul/ReadVariableOp2Z
+model_1/conv_s_n2d_9/Reshape/ReadVariableOp+model_1/conv_s_n2d_9/Reshape/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
╚
g
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_1473344

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         └2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
з
њ
.__inference_conv_s_n2d_8_layer_call_fn_1475667

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14728552
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┤
M
1__inference_up_sampling2d_1_layer_call_fn_1472507

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_14725012
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
кЂ
┼
D__inference_model_1_layer_call_and_return_conditional_losses_1474747
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource*
&dense_2_matmul_readvariableop_resource*
&dense_3_matmul_readvariableop_resource*
&dense_5_matmul_readvariableop_resource*
&dense_4_matmul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource0
,conv_s_n2d_5_reshape_readvariableop_resource/
+conv_s_n2d_5_matmul_readvariableop_resource0
,conv_s_n2d_5_biasadd_readvariableop_resource0
,conv_s_n2d_6_reshape_readvariableop_resource/
+conv_s_n2d_6_matmul_readvariableop_resource0
,conv_s_n2d_6_biasadd_readvariableop_resource0
,conv_s_n2d_7_reshape_readvariableop_resource/
+conv_s_n2d_7_matmul_readvariableop_resource0
,conv_s_n2d_7_biasadd_readvariableop_resource0
,conv_s_n2d_8_reshape_readvariableop_resource/
+conv_s_n2d_8_matmul_readvariableop_resource0
,conv_s_n2d_8_biasadd_readvariableop_resource0
,conv_s_n2d_9_reshape_readvariableop_resource/
+conv_s_n2d_9_matmul_readvariableop_resource0
,conv_s_n2d_9_biasadd_readvariableop_resource
identityѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб#conv_s_n2d_5/BiasAdd/ReadVariableOpб"conv_s_n2d_5/MatMul/ReadVariableOpб#conv_s_n2d_5/Reshape/ReadVariableOpб#conv_s_n2d_6/BiasAdd/ReadVariableOpб"conv_s_n2d_6/MatMul/ReadVariableOpб#conv_s_n2d_6/Reshape/ReadVariableOpб#conv_s_n2d_7/BiasAdd/ReadVariableOpб"conv_s_n2d_7/MatMul/ReadVariableOpб#conv_s_n2d_7/Reshape/ReadVariableOpб#conv_s_n2d_8/BiasAdd/ReadVariableOpб"conv_s_n2d_8/MatMul/ReadVariableOpб#conv_s_n2d_8/Reshape/ReadVariableOpб#conv_s_n2d_9/BiasAdd/ReadVariableOpб"conv_s_n2d_9/MatMul/ReadVariableOpб#conv_s_n2d_9/Reshape/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/MatMul/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/MatMul/ReadVariableOpЦ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpЇ
dense_1/MatMulMatMulinputs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulё
leaky_re_lu_10/LeakyRelu	LeakyReludense_1/MatMul:product:0*'
_output_shapes
:          2
leaky_re_lu_10/LeakyReluЦ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_2/MatMul/ReadVariableOpФ
dense_2/MatMulMatMul&leaky_re_lu_10/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/MatMulё
leaky_re_lu_11/LeakyRelu	LeakyReludense_2/MatMul:product:0*'
_output_shapes
:         @2
leaky_re_lu_11/LeakyReluд
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
dense_3/MatMul/ReadVariableOpг
dense_3/MatMulMatMul&leaky_re_lu_11/LeakyRelu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/MatMulЁ
leaky_re_lu_12/LeakyRelu	LeakyReludense_3/MatMul:product:0*(
_output_shapes
:         ђ2
leaky_re_lu_12/LeakyReluд
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	d└*
dtype02
dense_5/MatMul/ReadVariableOpј
dense_5/MatMulMatMulinputs_1%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
dense_5/MatMulД
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_4/MatMul/ReadVariableOpг
dense_4/MatMulMatMul&leaky_re_lu_12/LeakyRelu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/MatMul¤
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/y┘
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization/batchnorm/Rsqrt█
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpо
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/mul┼
#batch_normalization/batchnorm/mul_1Muldense_5/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └2%
#batch_normalization/batchnorm/mul_1Н
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1о
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization/batchnorm/mul_2Н
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2н
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2%
#batch_normalization/batchnorm/add_1Ё
leaky_re_lu_13/LeakyRelu	LeakyReludense_4/MatMul:product:0*(
_output_shapes
:         ђ2
leaky_re_lu_13/LeakyReluћ
leaky_re_lu_14/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         └2
leaky_re_lu_14/LeakyRelux
reshape_2/ShapeShape&leaky_re_lu_14/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shapeѕ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackї
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1ї
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2ъ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3Ш
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shapeх
reshape_2/ReshapeReshape&leaky_re_lu_14/LeakyRelu:activations:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
reshape_2/Reshapex
reshape_1/ShapeShape&leaky_re_lu_13/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shapeѕ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackї
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1ї
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2ъ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3Ш
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeх
reshape_1/ReshapeReshape&leaky_re_lu_13/LeakyRelu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
reshape_1/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisО
concatenate_2/concatConcatV2reshape_2/Reshape:output:0reshape_1/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:          >2
concatenate_2/concat└
#conv_s_n2d_5/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_5_reshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02%
#conv_s_n2d_5/Reshape/ReadVariableOpЅ
conv_s_n2d_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
conv_s_n2d_5/Reshape/shape│
conv_s_n2d_5/ReshapeReshape+conv_s_n2d_5/Reshape/ReadVariableOp:value:0#conv_s_n2d_5/Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2
conv_s_n2d_5/ReshapeІ
conv_s_n2d_5/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/transpose/permг
conv_s_n2d_5/transpose	Transposeconv_s_n2d_5/Reshape:output:0$conv_s_n2d_5/transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
conv_s_n2d_5/transposeх
"conv_s_n2d_5/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02$
"conv_s_n2d_5/MatMul/ReadVariableOpЦ
conv_s_n2d_5/MatMulMatMul*conv_s_n2d_5/MatMul/ReadVariableOp:value:0conv_s_n2d_5/transpose:y:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/MatMulm
conv_s_n2d_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_5/pow/yљ
conv_s_n2d_5/powPowconv_s_n2d_5/MatMul:product:0conv_s_n2d_5/pow/y:output:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/powy
conv_s_n2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/Const
conv_s_n2d_5/SumSumconv_s_n2d_5/pow:z:0conv_s_n2d_5/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/Sumq
conv_s_n2d_5/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_5/pow_1/yі
conv_s_n2d_5/pow_1Powconv_s_n2d_5/Sum:output:0conv_s_n2d_5/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/pow_1m
conv_s_n2d_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_5/add/yЃ
conv_s_n2d_5/addAddV2conv_s_n2d_5/pow_1:z:0conv_s_n2d_5/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/addЋ
conv_s_n2d_5/truedivRealDivconv_s_n2d_5/MatMul:product:0conv_s_n2d_5/add:z:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/truedivЏ
conv_s_n2d_5/MatMul_1MatMulconv_s_n2d_5/truediv:z:0conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/MatMul_1q
conv_s_n2d_5/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_5/pow_2/yЎ
conv_s_n2d_5/pow_2Powconv_s_n2d_5/MatMul_1:product:0conv_s_n2d_5/pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/pow_2}
conv_s_n2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/Const_1Є
conv_s_n2d_5/Sum_1Sumconv_s_n2d_5/pow_2:z:0conv_s_n2d_5/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/Sum_1q
conv_s_n2d_5/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_5/pow_3/yї
conv_s_n2d_5/pow_3Powconv_s_n2d_5/Sum_1:output:0conv_s_n2d_5/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/pow_3q
conv_s_n2d_5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_5/add_1/yЅ
conv_s_n2d_5/add_1AddV2conv_s_n2d_5/pow_3:z:0conv_s_n2d_5/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/add_1ъ
conv_s_n2d_5/truediv_1RealDivconv_s_n2d_5/MatMul_1:product:0conv_s_n2d_5/add_1:z:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/truediv_1Џ
conv_s_n2d_5/MatMul_2MatMulconv_s_n2d_5/truediv:z:0conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/MatMul_2Ј
conv_s_n2d_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/transpose_1/perm»
conv_s_n2d_5/transpose_1	Transposeconv_s_n2d_5/truediv_1:z:0&conv_s_n2d_5/transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/transpose_1а
conv_s_n2d_5/MatMul_3MatMulconv_s_n2d_5/MatMul_2:product:0conv_s_n2d_5/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_5/MatMul_3Ц
conv_s_n2d_5/truediv_2RealDivconv_s_n2d_5/Reshape:output:0conv_s_n2d_5/MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
conv_s_n2d_5/truediv_2Ћ
conv_s_n2d_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
conv_s_n2d_5/Reshape_1/shape░
conv_s_n2d_5/Reshape_1Reshapeconv_s_n2d_5/truediv_2:z:0%conv_s_n2d_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
conv_s_n2d_5/Reshape_1р
conv_s_n2d_5/convolutionConv2Dconcatenate_2/concat:output:0conv_s_n2d_5/Reshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
conv_s_n2d_5/convolution┤
#conv_s_n2d_5/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_5_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02%
#conv_s_n2d_5/BiasAdd/ReadVariableOp┬
conv_s_n2d_5/BiasAddBiasAdd!conv_s_n2d_5/convolution:output:0+conv_s_n2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2
conv_s_n2d_5/BiasAddw
up_sampling2d/ShapeShapeconv_s_n2d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d/Shapeљ
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackћ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1ћ
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2б
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Constќ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulЧ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_5/BiasAdd:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:          >ђ*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor░
leaky_re_lu_15/LeakyRelu	LeakyRelu;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*0
_output_shapes
:          >ђ2
leaky_re_lu_15/LeakyRelu└
#conv_s_n2d_6/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_6_reshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02%
#conv_s_n2d_6/Reshape/ReadVariableOpЅ
conv_s_n2d_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
conv_s_n2d_6/Reshape/shape│
conv_s_n2d_6/ReshapeReshape+conv_s_n2d_6/Reshape/ReadVariableOp:value:0#conv_s_n2d_6/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2
conv_s_n2d_6/ReshapeІ
conv_s_n2d_6/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/transpose/permг
conv_s_n2d_6/transpose	Transposeconv_s_n2d_6/Reshape:output:0$conv_s_n2d_6/transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
conv_s_n2d_6/transpose┤
"conv_s_n2d_6/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02$
"conv_s_n2d_6/MatMul/ReadVariableOpд
conv_s_n2d_6/MatMulMatMul*conv_s_n2d_6/MatMul/ReadVariableOp:value:0conv_s_n2d_6/transpose:y:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/MatMulm
conv_s_n2d_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_6/pow/yЉ
conv_s_n2d_6/powPowconv_s_n2d_6/MatMul:product:0conv_s_n2d_6/pow/y:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/powy
conv_s_n2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/Const
conv_s_n2d_6/SumSumconv_s_n2d_6/pow:z:0conv_s_n2d_6/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/Sumq
conv_s_n2d_6/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_6/pow_1/yі
conv_s_n2d_6/pow_1Powconv_s_n2d_6/Sum:output:0conv_s_n2d_6/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/pow_1m
conv_s_n2d_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_6/add/yЃ
conv_s_n2d_6/addAddV2conv_s_n2d_6/pow_1:z:0conv_s_n2d_6/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/addќ
conv_s_n2d_6/truedivRealDivconv_s_n2d_6/MatMul:product:0conv_s_n2d_6/add:z:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/truedivџ
conv_s_n2d_6/MatMul_1MatMulconv_s_n2d_6/truediv:z:0conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/MatMul_1q
conv_s_n2d_6/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_6/pow_2/yў
conv_s_n2d_6/pow_2Powconv_s_n2d_6/MatMul_1:product:0conv_s_n2d_6/pow_2/y:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/pow_2}
conv_s_n2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/Const_1Є
conv_s_n2d_6/Sum_1Sumconv_s_n2d_6/pow_2:z:0conv_s_n2d_6/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/Sum_1q
conv_s_n2d_6/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_6/pow_3/yї
conv_s_n2d_6/pow_3Powconv_s_n2d_6/Sum_1:output:0conv_s_n2d_6/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/pow_3q
conv_s_n2d_6/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_6/add_1/yЅ
conv_s_n2d_6/add_1AddV2conv_s_n2d_6/pow_3:z:0conv_s_n2d_6/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/add_1Ю
conv_s_n2d_6/truediv_1RealDivconv_s_n2d_6/MatMul_1:product:0conv_s_n2d_6/add_1:z:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/truediv_1џ
conv_s_n2d_6/MatMul_2MatMulconv_s_n2d_6/truediv:z:0conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/MatMul_2Ј
conv_s_n2d_6/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/transpose_1/perm«
conv_s_n2d_6/transpose_1	Transposeconv_s_n2d_6/truediv_1:z:0&conv_s_n2d_6/transpose_1/perm:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/transpose_1а
conv_s_n2d_6/MatMul_3MatMulconv_s_n2d_6/MatMul_2:product:0conv_s_n2d_6/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_6/MatMul_3Ц
conv_s_n2d_6/truediv_2RealDivconv_s_n2d_6/Reshape:output:0conv_s_n2d_6/MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
conv_s_n2d_6/truediv_2Ћ
conv_s_n2d_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
conv_s_n2d_6/Reshape_1/shape░
conv_s_n2d_6/Reshape_1Reshapeconv_s_n2d_6/truediv_2:z:0%conv_s_n2d_6/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
conv_s_n2d_6/Reshape_1ж
conv_s_n2d_6/convolutionConv2D&leaky_re_lu_15/LeakyRelu:activations:0conv_s_n2d_6/Reshape_1:output:0*
T0*/
_output_shapes
:          >@*
paddingSAME*
strides
2
conv_s_n2d_6/convolution│
#conv_s_n2d_6/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv_s_n2d_6/BiasAdd/ReadVariableOp┴
conv_s_n2d_6/BiasAddBiasAdd!conv_s_n2d_6/convolution:output:0+conv_s_n2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          >@2
conv_s_n2d_6/BiasAdd{
up_sampling2d_1/ShapeShapeconv_s_n2d_6/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shapeћ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackў
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1ў
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2«
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Constъ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЂ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_6/BiasAdd:output:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:         @|@*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor▒
leaky_re_lu_16/LeakyRelu	LeakyRelu=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*/
_output_shapes
:         @|@2
leaky_re_lu_16/LeakyRelu┐
#conv_s_n2d_7/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_7_reshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02%
#conv_s_n2d_7/Reshape/ReadVariableOpЅ
conv_s_n2d_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
conv_s_n2d_7/Reshape/shape│
conv_s_n2d_7/ReshapeReshape+conv_s_n2d_7/Reshape/ReadVariableOp:value:0#conv_s_n2d_7/Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2
conv_s_n2d_7/ReshapeІ
conv_s_n2d_7/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/transpose/permг
conv_s_n2d_7/transpose	Transposeconv_s_n2d_7/Reshape:output:0$conv_s_n2d_7/transpose/perm:output:0*
T0*
_output_shapes
:	 └2
conv_s_n2d_7/transpose┤
"conv_s_n2d_7/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"conv_s_n2d_7/MatMul/ReadVariableOpд
conv_s_n2d_7/MatMulMatMul*conv_s_n2d_7/MatMul/ReadVariableOp:value:0conv_s_n2d_7/transpose:y:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/MatMulm
conv_s_n2d_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_7/pow/yЉ
conv_s_n2d_7/powPowconv_s_n2d_7/MatMul:product:0conv_s_n2d_7/pow/y:output:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/powy
conv_s_n2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/Const
conv_s_n2d_7/SumSumconv_s_n2d_7/pow:z:0conv_s_n2d_7/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/Sumq
conv_s_n2d_7/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_7/pow_1/yі
conv_s_n2d_7/pow_1Powconv_s_n2d_7/Sum:output:0conv_s_n2d_7/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/pow_1m
conv_s_n2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_7/add/yЃ
conv_s_n2d_7/addAddV2conv_s_n2d_7/pow_1:z:0conv_s_n2d_7/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/addќ
conv_s_n2d_7/truedivRealDivconv_s_n2d_7/MatMul:product:0conv_s_n2d_7/add:z:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/truedivџ
conv_s_n2d_7/MatMul_1MatMulconv_s_n2d_7/truediv:z:0conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/MatMul_1q
conv_s_n2d_7/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_7/pow_2/yў
conv_s_n2d_7/pow_2Powconv_s_n2d_7/MatMul_1:product:0conv_s_n2d_7/pow_2/y:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/pow_2}
conv_s_n2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/Const_1Є
conv_s_n2d_7/Sum_1Sumconv_s_n2d_7/pow_2:z:0conv_s_n2d_7/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/Sum_1q
conv_s_n2d_7/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_7/pow_3/yї
conv_s_n2d_7/pow_3Powconv_s_n2d_7/Sum_1:output:0conv_s_n2d_7/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/pow_3q
conv_s_n2d_7/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_7/add_1/yЅ
conv_s_n2d_7/add_1AddV2conv_s_n2d_7/pow_3:z:0conv_s_n2d_7/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/add_1Ю
conv_s_n2d_7/truediv_1RealDivconv_s_n2d_7/MatMul_1:product:0conv_s_n2d_7/add_1:z:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/truediv_1џ
conv_s_n2d_7/MatMul_2MatMulconv_s_n2d_7/truediv:z:0conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/MatMul_2Ј
conv_s_n2d_7/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/transpose_1/perm«
conv_s_n2d_7/transpose_1	Transposeconv_s_n2d_7/truediv_1:z:0&conv_s_n2d_7/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/transpose_1а
conv_s_n2d_7/MatMul_3MatMulconv_s_n2d_7/MatMul_2:product:0conv_s_n2d_7/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_7/MatMul_3Ц
conv_s_n2d_7/truediv_2RealDivconv_s_n2d_7/Reshape:output:0conv_s_n2d_7/MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
conv_s_n2d_7/truediv_2Ћ
conv_s_n2d_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       2
conv_s_n2d_7/Reshape_1/shape»
conv_s_n2d_7/Reshape_1Reshapeconv_s_n2d_7/truediv_2:z:0%conv_s_n2d_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
conv_s_n2d_7/Reshape_1ж
conv_s_n2d_7/convolutionConv2D&leaky_re_lu_16/LeakyRelu:activations:0conv_s_n2d_7/Reshape_1:output:0*
T0*/
_output_shapes
:         @| *
paddingSAME*
strides
2
conv_s_n2d_7/convolution│
#conv_s_n2d_7/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv_s_n2d_7/BiasAdd/ReadVariableOp┴
conv_s_n2d_7/BiasAddBiasAdd!conv_s_n2d_7/convolution:output:0+conv_s_n2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @| 2
conv_s_n2d_7/BiasAdd{
up_sampling2d_2/ShapeShapeconv_s_n2d_7/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЃ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_7/BiasAdd:output:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђЭ *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor│
leaky_re_lu_17/LeakyRelu	LeakyRelu=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ 2
leaky_re_lu_17/LeakyRelu┐
#conv_s_n2d_8/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_8_reshape_readvariableop_resource*&
_output_shapes
: *
dtype02%
#conv_s_n2d_8/Reshape/ReadVariableOpЅ
conv_s_n2d_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Reshape/shape│
conv_s_n2d_8/ReshapeReshape+conv_s_n2d_8/Reshape/ReadVariableOp:value:0#conv_s_n2d_8/Reshape/shape:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/ReshapeІ
conv_s_n2d_8/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/transpose/permг
conv_s_n2d_8/transpose	Transposeconv_s_n2d_8/Reshape:output:0$conv_s_n2d_8/transpose/perm:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/transpose┤
"conv_s_n2d_8/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"conv_s_n2d_8/MatMul/ReadVariableOpд
conv_s_n2d_8/MatMulMatMul*conv_s_n2d_8/MatMul/ReadVariableOp:value:0conv_s_n2d_8/transpose:y:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/MatMulm
conv_s_n2d_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_8/pow/yЉ
conv_s_n2d_8/powPowconv_s_n2d_8/MatMul:product:0conv_s_n2d_8/pow/y:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/powy
conv_s_n2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Const
conv_s_n2d_8/SumSumconv_s_n2d_8/pow:z:0conv_s_n2d_8/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/Sumq
conv_s_n2d_8/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_8/pow_1/yі
conv_s_n2d_8/pow_1Powconv_s_n2d_8/Sum:output:0conv_s_n2d_8/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/pow_1m
conv_s_n2d_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_8/add/yЃ
conv_s_n2d_8/addAddV2conv_s_n2d_8/pow_1:z:0conv_s_n2d_8/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/addќ
conv_s_n2d_8/truedivRealDivconv_s_n2d_8/MatMul:product:0conv_s_n2d_8/add:z:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/truedivџ
conv_s_n2d_8/MatMul_1MatMulconv_s_n2d_8/truediv:z:0conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_1q
conv_s_n2d_8/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_8/pow_2/yў
conv_s_n2d_8/pow_2Powconv_s_n2d_8/MatMul_1:product:0conv_s_n2d_8/pow_2/y:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/pow_2}
conv_s_n2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Const_1Є
conv_s_n2d_8/Sum_1Sumconv_s_n2d_8/pow_2:z:0conv_s_n2d_8/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/Sum_1q
conv_s_n2d_8/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_8/pow_3/yї
conv_s_n2d_8/pow_3Powconv_s_n2d_8/Sum_1:output:0conv_s_n2d_8/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/pow_3q
conv_s_n2d_8/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_8/add_1/yЅ
conv_s_n2d_8/add_1AddV2conv_s_n2d_8/pow_3:z:0conv_s_n2d_8/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/add_1Ю
conv_s_n2d_8/truediv_1RealDivconv_s_n2d_8/MatMul_1:product:0conv_s_n2d_8/add_1:z:0*
T0*
_output_shapes

:2
conv_s_n2d_8/truediv_1џ
conv_s_n2d_8/MatMul_2MatMulconv_s_n2d_8/truediv:z:0conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_2Ј
conv_s_n2d_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/transpose_1/perm«
conv_s_n2d_8/transpose_1	Transposeconv_s_n2d_8/truediv_1:z:0&conv_s_n2d_8/transpose_1/perm:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/transpose_1а
conv_s_n2d_8/MatMul_3MatMulconv_s_n2d_8/MatMul_2:product:0conv_s_n2d_8/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_3Ц
conv_s_n2d_8/truediv_2RealDivconv_s_n2d_8/Reshape:output:0conv_s_n2d_8/MatMul_3:product:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/truediv_2Ћ
conv_s_n2d_8/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv_s_n2d_8/Reshape_1/shape»
conv_s_n2d_8/Reshape_1Reshapeconv_s_n2d_8/truediv_2:z:0%conv_s_n2d_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
conv_s_n2d_8/Reshape_1в
conv_s_n2d_8/convolutionConv2D&leaky_re_lu_17/LeakyRelu:activations:0conv_s_n2d_8/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2
conv_s_n2d_8/convolution│
#conv_s_n2d_8/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#conv_s_n2d_8/BiasAdd/ReadVariableOp├
conv_s_n2d_8/BiasAddBiasAdd!conv_s_n2d_8/convolution:output:0+conv_s_n2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
conv_s_n2d_8/BiasAdd{
up_sampling2d_3/ShapeShapeconv_s_n2d_8/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shapeћ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackў
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1ў
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2«
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Constъ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulЃ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_8/BiasAdd:output:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor│
leaky_re_lu_18/LeakyRelu	LeakyRelu=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2
leaky_re_lu_18/LeakyRelu┐
#conv_s_n2d_9/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_9_reshape_readvariableop_resource*&
_output_shapes
:*
dtype02%
#conv_s_n2d_9/Reshape/ReadVariableOpЅ
conv_s_n2d_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Reshape/shape│
conv_s_n2d_9/ReshapeReshape+conv_s_n2d_9/Reshape/ReadVariableOp:value:0#conv_s_n2d_9/Reshape/shape:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/ReshapeІ
conv_s_n2d_9/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/transpose/permг
conv_s_n2d_9/transpose	Transposeconv_s_n2d_9/Reshape:output:0$conv_s_n2d_9/transpose/perm:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/transpose┤
"conv_s_n2d_9/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"conv_s_n2d_9/MatMul/ReadVariableOpд
conv_s_n2d_9/MatMulMatMul*conv_s_n2d_9/MatMul/ReadVariableOp:value:0conv_s_n2d_9/transpose:y:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/MatMulm
conv_s_n2d_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_9/pow/yЉ
conv_s_n2d_9/powPowconv_s_n2d_9/MatMul:product:0conv_s_n2d_9/pow/y:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/powy
conv_s_n2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Const
conv_s_n2d_9/SumSumconv_s_n2d_9/pow:z:0conv_s_n2d_9/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/Sumq
conv_s_n2d_9/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_9/pow_1/yі
conv_s_n2d_9/pow_1Powconv_s_n2d_9/Sum:output:0conv_s_n2d_9/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/pow_1m
conv_s_n2d_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_9/add/yЃ
conv_s_n2d_9/addAddV2conv_s_n2d_9/pow_1:z:0conv_s_n2d_9/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/addќ
conv_s_n2d_9/truedivRealDivconv_s_n2d_9/MatMul:product:0conv_s_n2d_9/add:z:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/truedivџ
conv_s_n2d_9/MatMul_1MatMulconv_s_n2d_9/truediv:z:0conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_1q
conv_s_n2d_9/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_9/pow_2/yў
conv_s_n2d_9/pow_2Powconv_s_n2d_9/MatMul_1:product:0conv_s_n2d_9/pow_2/y:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/pow_2}
conv_s_n2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Const_1Є
conv_s_n2d_9/Sum_1Sumconv_s_n2d_9/pow_2:z:0conv_s_n2d_9/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/Sum_1q
conv_s_n2d_9/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_9/pow_3/yї
conv_s_n2d_9/pow_3Powconv_s_n2d_9/Sum_1:output:0conv_s_n2d_9/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/pow_3q
conv_s_n2d_9/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_9/add_1/yЅ
conv_s_n2d_9/add_1AddV2conv_s_n2d_9/pow_3:z:0conv_s_n2d_9/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/add_1Ю
conv_s_n2d_9/truediv_1RealDivconv_s_n2d_9/MatMul_1:product:0conv_s_n2d_9/add_1:z:0*
T0*
_output_shapes

:2
conv_s_n2d_9/truediv_1џ
conv_s_n2d_9/MatMul_2MatMulconv_s_n2d_9/truediv:z:0conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_2Ј
conv_s_n2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/transpose_1/perm«
conv_s_n2d_9/transpose_1	Transposeconv_s_n2d_9/truediv_1:z:0&conv_s_n2d_9/transpose_1/perm:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/transpose_1а
conv_s_n2d_9/MatMul_3MatMulconv_s_n2d_9/MatMul_2:product:0conv_s_n2d_9/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_3Ц
conv_s_n2d_9/truediv_2RealDivconv_s_n2d_9/Reshape:output:0conv_s_n2d_9/MatMul_3:product:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/truediv_2Ћ
conv_s_n2d_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
conv_s_n2d_9/Reshape_1/shape»
conv_s_n2d_9/Reshape_1Reshapeconv_s_n2d_9/truediv_2:z:0%conv_s_n2d_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
conv_s_n2d_9/Reshape_1в
conv_s_n2d_9/convolutionConv2D&leaky_re_lu_18/LeakyRelu:activations:0conv_s_n2d_9/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2
conv_s_n2d_9/convolution│
#conv_s_n2d_9/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#conv_s_n2d_9/BiasAdd/ReadVariableOp├
conv_s_n2d_9/BiasAddBiasAdd!conv_s_n2d_9/convolution:output:0+conv_s_n2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
conv_s_n2d_9/BiasAdd{
up_sampling2d_4/ShapeShapeconv_s_n2d_9/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shapeћ
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stackў
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1ў
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2«
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Constъ
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulЃ
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_9/BiasAdd:output:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor│
leaky_re_lu_19/LeakyRelu	LeakyRelu=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2
leaky_re_lu_19/LeakyReluЮ
IdentityIdentity&leaky_re_lu_19/LeakyRelu:activations:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp$^conv_s_n2d_5/BiasAdd/ReadVariableOp#^conv_s_n2d_5/MatMul/ReadVariableOp$^conv_s_n2d_5/Reshape/ReadVariableOp$^conv_s_n2d_6/BiasAdd/ReadVariableOp#^conv_s_n2d_6/MatMul/ReadVariableOp$^conv_s_n2d_6/Reshape/ReadVariableOp$^conv_s_n2d_7/BiasAdd/ReadVariableOp#^conv_s_n2d_7/MatMul/ReadVariableOp$^conv_s_n2d_7/Reshape/ReadVariableOp$^conv_s_n2d_8/BiasAdd/ReadVariableOp#^conv_s_n2d_8/MatMul/ReadVariableOp$^conv_s_n2d_8/Reshape/ReadVariableOp$^conv_s_n2d_9/BiasAdd/ReadVariableOp#^conv_s_n2d_9/MatMul/ReadVariableOp$^conv_s_n2d_9/Reshape/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*1
_output_shapes
:         ђЭ2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2J
#conv_s_n2d_5/BiasAdd/ReadVariableOp#conv_s_n2d_5/BiasAdd/ReadVariableOp2H
"conv_s_n2d_5/MatMul/ReadVariableOp"conv_s_n2d_5/MatMul/ReadVariableOp2J
#conv_s_n2d_5/Reshape/ReadVariableOp#conv_s_n2d_5/Reshape/ReadVariableOp2J
#conv_s_n2d_6/BiasAdd/ReadVariableOp#conv_s_n2d_6/BiasAdd/ReadVariableOp2H
"conv_s_n2d_6/MatMul/ReadVariableOp"conv_s_n2d_6/MatMul/ReadVariableOp2J
#conv_s_n2d_6/Reshape/ReadVariableOp#conv_s_n2d_6/Reshape/ReadVariableOp2J
#conv_s_n2d_7/BiasAdd/ReadVariableOp#conv_s_n2d_7/BiasAdd/ReadVariableOp2H
"conv_s_n2d_7/MatMul/ReadVariableOp"conv_s_n2d_7/MatMul/ReadVariableOp2J
#conv_s_n2d_7/Reshape/ReadVariableOp#conv_s_n2d_7/Reshape/ReadVariableOp2J
#conv_s_n2d_8/BiasAdd/ReadVariableOp#conv_s_n2d_8/BiasAdd/ReadVariableOp2H
"conv_s_n2d_8/MatMul/ReadVariableOp"conv_s_n2d_8/MatMul/ReadVariableOp2J
#conv_s_n2d_8/Reshape/ReadVariableOp#conv_s_n2d_8/Reshape/ReadVariableOp2J
#conv_s_n2d_9/BiasAdd/ReadVariableOp#conv_s_n2d_9/BiasAdd/ReadVariableOp2H
"conv_s_n2d_9/MatMul/ReadVariableOp"conv_s_n2d_9/MatMul/ReadVariableOp2J
#conv_s_n2d_9/Reshape/ReadVariableOp#conv_s_n2d_9/Reshape/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs/1
Є
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1472286

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
њ
L
0__inference_leaky_re_lu_15_layer_call_fn_1475334

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_14735332
PartitionedCallЄ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
К'
│
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475495

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	 └2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	└2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	└2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	└2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

: 2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

: 2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      @       2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
х9
й

 __inference__traced_save_1475902
file_prefix-
)savev2_dense_1_kernel_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop2
.savev2_conv_s_n2d_5_kernel_read_readvariableop0
,savev2_conv_s_n2d_5_bias_read_readvariableop.
*savev2_conv_s_n2d_5_sn_read_readvariableop2
.savev2_conv_s_n2d_6_kernel_read_readvariableop0
,savev2_conv_s_n2d_6_bias_read_readvariableop.
*savev2_conv_s_n2d_6_sn_read_readvariableop2
.savev2_conv_s_n2d_7_kernel_read_readvariableop0
,savev2_conv_s_n2d_7_bias_read_readvariableop.
*savev2_conv_s_n2d_7_sn_read_readvariableop2
.savev2_conv_s_n2d_8_kernel_read_readvariableop0
,savev2_conv_s_n2d_8_bias_read_readvariableop.
*savev2_conv_s_n2d_8_sn_read_readvariableop2
.savev2_conv_s_n2d_9_kernel_read_readvariableop0
,savev2_conv_s_n2d_9_bias_read_readvariableop.
*savev2_conv_s_n2d_9_sn_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЛ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с

value┘
Bо
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-6/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-7/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-8/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-9/sn/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-10/sn/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names║
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┬

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop)savev2_dense_2_kernel_read_readvariableop)savev2_dense_3_kernel_read_readvariableop)savev2_dense_5_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop.savev2_conv_s_n2d_5_kernel_read_readvariableop,savev2_conv_s_n2d_5_bias_read_readvariableop*savev2_conv_s_n2d_5_sn_read_readvariableop.savev2_conv_s_n2d_6_kernel_read_readvariableop,savev2_conv_s_n2d_6_bias_read_readvariableop*savev2_conv_s_n2d_6_sn_read_readvariableop.savev2_conv_s_n2d_7_kernel_read_readvariableop,savev2_conv_s_n2d_7_bias_read_readvariableop*savev2_conv_s_n2d_7_sn_read_readvariableop.savev2_conv_s_n2d_8_kernel_read_readvariableop,savev2_conv_s_n2d_8_bias_read_readvariableop*savev2_conv_s_n2d_8_sn_read_readvariableop.savev2_conv_s_n2d_9_kernel_read_readvariableop,savev2_conv_s_n2d_9_bias_read_readvariableop*savev2_conv_s_n2d_9_sn_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ў
_input_shapesЄ
ё: : : @:	@ђ:	d└:└:└:└:└:
ђђ:ђ:ђ:	ђ:ђ@:@:@:@ : : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: :$ 

_output_shapes

: @:%!

_output_shapes
:	@ђ:%!

_output_shapes
:	d└:!

_output_shapes	
:└:!

_output_shapes	
:└:!

_output_shapes	
:└:!

_output_shapes	
:└:&	"
 
_output_shapes
:
ђђ:-
)
'
_output_shapes
:ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ:-)
'
_output_shapes
:ђ@: 

_output_shapes
:@:$ 

_output_shapes

:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :$ 

_output_shapes

: :,(
&
_output_shapes
: : 

_output_shapes
::$ 

_output_shapes

::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
Е
L
0__inference_leaky_re_lu_14_layer_call_fn_1475047

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_14733442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
г
▀
%__inference_signature_wrapper_1474125
input_3
input_4
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђЭ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *+
f&R$
"__inference__wrapped_model_14719372
StatefulPartitionedCallў
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ђЭ2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
ы
Ъ
D__inference_dense_2_layer_call_and_return_conditional_losses_1473196

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0**
_input_shapes
:          :2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
К'
│
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1472640

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	 └2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	└2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	└2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	└2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

: 2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

: 2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      @       2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
К'
│
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1472855

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	а2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	а2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	а2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	а2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	а2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	а2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
«
њ
.__inference_conv_s_n2d_5_layer_call_fn_1475216

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734972
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
Ѕ
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1472716

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
К'
│
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1473070

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	љ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	љ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	љ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	љ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	љ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	љ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
њ
.__inference_conv_s_n2d_9_layer_call_fn_1475796

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14731242
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_1473705

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш&
│
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1473454

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1Б
convolutionConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpј
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2	
BiasAdd╩
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
╦%
а
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1472264

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1х
convolutionConv2DinputsReshape_1:output:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpа
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAdd╔
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Е
L
0__inference_leaky_re_lu_12_layer_call_fn_1474941

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_14732452
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы
Ъ
D__inference_dense_2_layer_call_and_return_conditional_losses_1474886

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0**
_input_shapes
:          :2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┴
o
)__inference_dense_1_layer_call_fn_1474869

inputs
unknown
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14731642
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Г
G
+__inference_reshape_2_layer_call_fn_1475076

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_14733662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
л'
│
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475259

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1х
convolutionConv2DinputsReshape_1:output:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpа
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAdd▄
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
њ
.__inference_conv_s_n2d_5_layer_call_fn_1475324

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14722642
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╣и
¤
D__inference_model_1_layer_call_and_return_conditional_losses_1474444
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource*
&dense_2_matmul_readvariableop_resource*
&dense_3_matmul_readvariableop_resource*
&dense_5_matmul_readvariableop_resource*
&dense_4_matmul_readvariableop_resource/
+batch_normalization_assignmovingavg_14741551
-batch_normalization_assignmovingavg_1_1474161=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource0
,conv_s_n2d_5_reshape_readvariableop_resource/
+conv_s_n2d_5_matmul_readvariableop_resource0
,conv_s_n2d_5_biasadd_readvariableop_resource0
,conv_s_n2d_6_reshape_readvariableop_resource/
+conv_s_n2d_6_matmul_readvariableop_resource0
,conv_s_n2d_6_biasadd_readvariableop_resource0
,conv_s_n2d_7_reshape_readvariableop_resource/
+conv_s_n2d_7_matmul_readvariableop_resource0
,conv_s_n2d_7_biasadd_readvariableop_resource0
,conv_s_n2d_8_reshape_readvariableop_resource/
+conv_s_n2d_8_matmul_readvariableop_resource0
,conv_s_n2d_8_biasadd_readvariableop_resource0
,conv_s_n2d_9_reshape_readvariableop_resource/
+conv_s_n2d_9_matmul_readvariableop_resource0
,conv_s_n2d_9_biasadd_readvariableop_resource
identityѕб7batch_normalization/AssignMovingAvg/AssignSubVariableOpб2batch_normalization/AssignMovingAvg/ReadVariableOpб9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpб4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpбconv_s_n2d_5/AssignVariableOpб#conv_s_n2d_5/BiasAdd/ReadVariableOpб"conv_s_n2d_5/MatMul/ReadVariableOpб#conv_s_n2d_5/Reshape/ReadVariableOpбconv_s_n2d_6/AssignVariableOpб#conv_s_n2d_6/BiasAdd/ReadVariableOpб"conv_s_n2d_6/MatMul/ReadVariableOpб#conv_s_n2d_6/Reshape/ReadVariableOpбconv_s_n2d_7/AssignVariableOpб#conv_s_n2d_7/BiasAdd/ReadVariableOpб"conv_s_n2d_7/MatMul/ReadVariableOpб#conv_s_n2d_7/Reshape/ReadVariableOpбconv_s_n2d_8/AssignVariableOpб#conv_s_n2d_8/BiasAdd/ReadVariableOpб"conv_s_n2d_8/MatMul/ReadVariableOpб#conv_s_n2d_8/Reshape/ReadVariableOpбconv_s_n2d_9/AssignVariableOpб#conv_s_n2d_9/BiasAdd/ReadVariableOpб"conv_s_n2d_9/MatMul/ReadVariableOpб#conv_s_n2d_9/Reshape/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/MatMul/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/MatMul/ReadVariableOpЦ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpЇ
dense_1/MatMulMatMulinputs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulё
leaky_re_lu_10/LeakyRelu	LeakyReludense_1/MatMul:product:0*'
_output_shapes
:          2
leaky_re_lu_10/LeakyReluЦ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_2/MatMul/ReadVariableOpФ
dense_2/MatMulMatMul&leaky_re_lu_10/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/MatMulё
leaky_re_lu_11/LeakyRelu	LeakyReludense_2/MatMul:product:0*'
_output_shapes
:         @2
leaky_re_lu_11/LeakyReluд
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02
dense_3/MatMul/ReadVariableOpг
dense_3/MatMulMatMul&leaky_re_lu_11/LeakyRelu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/MatMulЁ
leaky_re_lu_12/LeakyRelu	LeakyReludense_3/MatMul:product:0*(
_output_shapes
:         ђ2
leaky_re_lu_12/LeakyReluд
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	d└*
dtype02
dense_5/MatMul/ReadVariableOpј
dense_5/MatMulMatMulinputs_1%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
dense_5/MatMulД
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
dense_4/MatMul/ReadVariableOpг
dense_4/MatMulMatMul&leaky_re_lu_12/LeakyRelu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_4/MatMul▓
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indicesя
 batch_normalization/moments/meanMeandense_5/MatMul:product:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2"
 batch_normalization/moments/mean╣
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	└2*
(batch_normalization/moments/StopGradientз
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense_5/MatMul:product:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         └2/
-batch_normalization/moments/SquaredDifference║
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indicesЃ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1Ѕ
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/1474155*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2+
)batch_normalization/AssignMovingAvg/decayЛ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1474155*
_output_shapes	
:└*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpО
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/1474155*
_output_shapes	
:└2)
'batch_normalization/AssignMovingAvg/sub╬
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/1474155*
_output_shapes	
:└2)
'batch_normalization/AssignMovingAvg/mulЕ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1474155+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/1474155*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpЈ
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/1474161*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2-
+batch_normalization/AssignMovingAvg_1/decayО
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_assignmovingavg_1_1474161*
_output_shapes	
:└*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpр
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/1474161*
_output_shapes	
:└2+
)batch_normalization/AssignMovingAvg_1/subп
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/1474161*
_output_shapes	
:└2+
)batch_normalization/AssignMovingAvg_1/mulх
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_assignmovingavg_1_1474161-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/1474161*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yМ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization/batchnorm/Rsqrt█
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpо
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/mul┼
#batch_normalization/batchnorm/mul_1Muldense_5/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         └2%
#batch_normalization/batchnorm/mul_1╠
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:└2%
#batch_normalization/batchnorm/mul_2¤
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpм
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2%
#batch_normalization/batchnorm/add_1Ё
leaky_re_lu_13/LeakyRelu	LeakyReludense_4/MatMul:product:0*(
_output_shapes
:         ђ2
leaky_re_lu_13/LeakyReluћ
leaky_re_lu_14/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         └2
leaky_re_lu_14/LeakyRelux
reshape_2/ShapeShape&leaky_re_lu_14/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_2/Shapeѕ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackї
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1ї
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2ъ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3Ш
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shapeх
reshape_2/ReshapeReshape&leaky_re_lu_14/LeakyRelu:activations:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
reshape_2/Reshapex
reshape_1/ShapeShape&leaky_re_lu_13/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shapeѕ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackї
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1ї
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2ъ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3Ш
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeх
reshape_1/ReshapeReshape&leaky_re_lu_13/LeakyRelu:activations:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:          >2
reshape_1/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisО
concatenate_2/concatConcatV2reshape_2/Reshape:output:0reshape_1/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:          >2
concatenate_2/concat└
#conv_s_n2d_5/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_5_reshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02%
#conv_s_n2d_5/Reshape/ReadVariableOpЅ
conv_s_n2d_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
conv_s_n2d_5/Reshape/shape│
conv_s_n2d_5/ReshapeReshape+conv_s_n2d_5/Reshape/ReadVariableOp:value:0#conv_s_n2d_5/Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2
conv_s_n2d_5/ReshapeІ
conv_s_n2d_5/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/transpose/permг
conv_s_n2d_5/transpose	Transposeconv_s_n2d_5/Reshape:output:0$conv_s_n2d_5/transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
conv_s_n2d_5/transposeх
"conv_s_n2d_5/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_5_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02$
"conv_s_n2d_5/MatMul/ReadVariableOpЦ
conv_s_n2d_5/MatMulMatMul*conv_s_n2d_5/MatMul/ReadVariableOp:value:0conv_s_n2d_5/transpose:y:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/MatMulm
conv_s_n2d_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_5/pow/yљ
conv_s_n2d_5/powPowconv_s_n2d_5/MatMul:product:0conv_s_n2d_5/pow/y:output:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/powy
conv_s_n2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/Const
conv_s_n2d_5/SumSumconv_s_n2d_5/pow:z:0conv_s_n2d_5/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/Sumq
conv_s_n2d_5/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_5/pow_1/yі
conv_s_n2d_5/pow_1Powconv_s_n2d_5/Sum:output:0conv_s_n2d_5/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/pow_1m
conv_s_n2d_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_5/add/yЃ
conv_s_n2d_5/addAddV2conv_s_n2d_5/pow_1:z:0conv_s_n2d_5/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/addЋ
conv_s_n2d_5/truedivRealDivconv_s_n2d_5/MatMul:product:0conv_s_n2d_5/add:z:0*
T0*
_output_shapes

:K2
conv_s_n2d_5/truedivЏ
conv_s_n2d_5/MatMul_1MatMulconv_s_n2d_5/truediv:z:0conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/MatMul_1q
conv_s_n2d_5/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_5/pow_2/yЎ
conv_s_n2d_5/pow_2Powconv_s_n2d_5/MatMul_1:product:0conv_s_n2d_5/pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/pow_2}
conv_s_n2d_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/Const_1Є
conv_s_n2d_5/Sum_1Sumconv_s_n2d_5/pow_2:z:0conv_s_n2d_5/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/Sum_1q
conv_s_n2d_5/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_5/pow_3/yї
conv_s_n2d_5/pow_3Powconv_s_n2d_5/Sum_1:output:0conv_s_n2d_5/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/pow_3q
conv_s_n2d_5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_5/add_1/yЅ
conv_s_n2d_5/add_1AddV2conv_s_n2d_5/pow_3:z:0conv_s_n2d_5/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_5/add_1ъ
conv_s_n2d_5/truediv_1RealDivconv_s_n2d_5/MatMul_1:product:0conv_s_n2d_5/add_1:z:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/truediv_1Џ
conv_s_n2d_5/MatMul_2MatMulconv_s_n2d_5/truediv:z:0conv_s_n2d_5/Reshape:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/MatMul_2Ј
conv_s_n2d_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_5/transpose_1/perm»
conv_s_n2d_5/transpose_1	Transposeconv_s_n2d_5/truediv_1:z:0&conv_s_n2d_5/transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_5/transpose_1а
conv_s_n2d_5/MatMul_3MatMulconv_s_n2d_5/MatMul_2:product:0conv_s_n2d_5/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_5/MatMul_3Ц
conv_s_n2d_5/truediv_2RealDivconv_s_n2d_5/Reshape:output:0conv_s_n2d_5/MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
conv_s_n2d_5/truediv_2с
conv_s_n2d_5/AssignVariableOpAssignVariableOp+conv_s_n2d_5_matmul_readvariableop_resourceconv_s_n2d_5/truediv_1:z:0#^conv_s_n2d_5/MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
conv_s_n2d_5/AssignVariableOpх
conv_s_n2d_5/Reshape_1/shapeConst^conv_s_n2d_5/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
conv_s_n2d_5/Reshape_1/shape░
conv_s_n2d_5/Reshape_1Reshapeconv_s_n2d_5/truediv_2:z:0%conv_s_n2d_5/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
conv_s_n2d_5/Reshape_1р
conv_s_n2d_5/convolutionConv2Dconcatenate_2/concat:output:0conv_s_n2d_5/Reshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
conv_s_n2d_5/convolution┤
#conv_s_n2d_5/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_5_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02%
#conv_s_n2d_5/BiasAdd/ReadVariableOp┬
conv_s_n2d_5/BiasAddBiasAdd!conv_s_n2d_5/convolution:output:0+conv_s_n2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2
conv_s_n2d_5/BiasAddw
up_sampling2d/ShapeShapeconv_s_n2d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d/Shapeљ
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stackћ
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1ћ
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2б
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Constќ
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mulЧ
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_5/BiasAdd:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:          >ђ*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor░
leaky_re_lu_15/LeakyRelu	LeakyRelu;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*0
_output_shapes
:          >ђ2
leaky_re_lu_15/LeakyRelu└
#conv_s_n2d_6/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_6_reshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02%
#conv_s_n2d_6/Reshape/ReadVariableOpЅ
conv_s_n2d_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
conv_s_n2d_6/Reshape/shape│
conv_s_n2d_6/ReshapeReshape+conv_s_n2d_6/Reshape/ReadVariableOp:value:0#conv_s_n2d_6/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2
conv_s_n2d_6/ReshapeІ
conv_s_n2d_6/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/transpose/permг
conv_s_n2d_6/transpose	Transposeconv_s_n2d_6/Reshape:output:0$conv_s_n2d_6/transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
conv_s_n2d_6/transpose┤
"conv_s_n2d_6/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02$
"conv_s_n2d_6/MatMul/ReadVariableOpд
conv_s_n2d_6/MatMulMatMul*conv_s_n2d_6/MatMul/ReadVariableOp:value:0conv_s_n2d_6/transpose:y:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/MatMulm
conv_s_n2d_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_6/pow/yЉ
conv_s_n2d_6/powPowconv_s_n2d_6/MatMul:product:0conv_s_n2d_6/pow/y:output:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/powy
conv_s_n2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/Const
conv_s_n2d_6/SumSumconv_s_n2d_6/pow:z:0conv_s_n2d_6/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/Sumq
conv_s_n2d_6/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_6/pow_1/yі
conv_s_n2d_6/pow_1Powconv_s_n2d_6/Sum:output:0conv_s_n2d_6/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/pow_1m
conv_s_n2d_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_6/add/yЃ
conv_s_n2d_6/addAddV2conv_s_n2d_6/pow_1:z:0conv_s_n2d_6/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/addќ
conv_s_n2d_6/truedivRealDivconv_s_n2d_6/MatMul:product:0conv_s_n2d_6/add:z:0*
T0*
_output_shapes
:	ђ2
conv_s_n2d_6/truedivџ
conv_s_n2d_6/MatMul_1MatMulconv_s_n2d_6/truediv:z:0conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/MatMul_1q
conv_s_n2d_6/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_6/pow_2/yў
conv_s_n2d_6/pow_2Powconv_s_n2d_6/MatMul_1:product:0conv_s_n2d_6/pow_2/y:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/pow_2}
conv_s_n2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/Const_1Є
conv_s_n2d_6/Sum_1Sumconv_s_n2d_6/pow_2:z:0conv_s_n2d_6/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/Sum_1q
conv_s_n2d_6/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_6/pow_3/yї
conv_s_n2d_6/pow_3Powconv_s_n2d_6/Sum_1:output:0conv_s_n2d_6/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/pow_3q
conv_s_n2d_6/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_6/add_1/yЅ
conv_s_n2d_6/add_1AddV2conv_s_n2d_6/pow_3:z:0conv_s_n2d_6/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_6/add_1Ю
conv_s_n2d_6/truediv_1RealDivconv_s_n2d_6/MatMul_1:product:0conv_s_n2d_6/add_1:z:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/truediv_1џ
conv_s_n2d_6/MatMul_2MatMulconv_s_n2d_6/truediv:z:0conv_s_n2d_6/Reshape:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/MatMul_2Ј
conv_s_n2d_6/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_6/transpose_1/perm«
conv_s_n2d_6/transpose_1	Transposeconv_s_n2d_6/truediv_1:z:0&conv_s_n2d_6/transpose_1/perm:output:0*
T0*
_output_shapes

:@2
conv_s_n2d_6/transpose_1а
conv_s_n2d_6/MatMul_3MatMulconv_s_n2d_6/MatMul_2:product:0conv_s_n2d_6/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_6/MatMul_3Ц
conv_s_n2d_6/truediv_2RealDivconv_s_n2d_6/Reshape:output:0conv_s_n2d_6/MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
conv_s_n2d_6/truediv_2с
conv_s_n2d_6/AssignVariableOpAssignVariableOp+conv_s_n2d_6_matmul_readvariableop_resourceconv_s_n2d_6/truediv_1:z:0#^conv_s_n2d_6/MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
conv_s_n2d_6/AssignVariableOpх
conv_s_n2d_6/Reshape_1/shapeConst^conv_s_n2d_6/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
conv_s_n2d_6/Reshape_1/shape░
conv_s_n2d_6/Reshape_1Reshapeconv_s_n2d_6/truediv_2:z:0%conv_s_n2d_6/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
conv_s_n2d_6/Reshape_1ж
conv_s_n2d_6/convolutionConv2D&leaky_re_lu_15/LeakyRelu:activations:0conv_s_n2d_6/Reshape_1:output:0*
T0*/
_output_shapes
:          >@*
paddingSAME*
strides
2
conv_s_n2d_6/convolution│
#conv_s_n2d_6/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#conv_s_n2d_6/BiasAdd/ReadVariableOp┴
conv_s_n2d_6/BiasAddBiasAdd!conv_s_n2d_6/convolution:output:0+conv_s_n2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          >@2
conv_s_n2d_6/BiasAdd{
up_sampling2d_1/ShapeShapeconv_s_n2d_6/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shapeћ
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stackў
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1ў
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2«
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Constъ
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mulЂ
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_6/BiasAdd:output:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:         @|@*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor▒
leaky_re_lu_16/LeakyRelu	LeakyRelu=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0*/
_output_shapes
:         @|@2
leaky_re_lu_16/LeakyRelu┐
#conv_s_n2d_7/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_7_reshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02%
#conv_s_n2d_7/Reshape/ReadVariableOpЅ
conv_s_n2d_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
conv_s_n2d_7/Reshape/shape│
conv_s_n2d_7/ReshapeReshape+conv_s_n2d_7/Reshape/ReadVariableOp:value:0#conv_s_n2d_7/Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2
conv_s_n2d_7/ReshapeІ
conv_s_n2d_7/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/transpose/permг
conv_s_n2d_7/transpose	Transposeconv_s_n2d_7/Reshape:output:0$conv_s_n2d_7/transpose/perm:output:0*
T0*
_output_shapes
:	 └2
conv_s_n2d_7/transpose┤
"conv_s_n2d_7/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"conv_s_n2d_7/MatMul/ReadVariableOpд
conv_s_n2d_7/MatMulMatMul*conv_s_n2d_7/MatMul/ReadVariableOp:value:0conv_s_n2d_7/transpose:y:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/MatMulm
conv_s_n2d_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_7/pow/yЉ
conv_s_n2d_7/powPowconv_s_n2d_7/MatMul:product:0conv_s_n2d_7/pow/y:output:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/powy
conv_s_n2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/Const
conv_s_n2d_7/SumSumconv_s_n2d_7/pow:z:0conv_s_n2d_7/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/Sumq
conv_s_n2d_7/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_7/pow_1/yі
conv_s_n2d_7/pow_1Powconv_s_n2d_7/Sum:output:0conv_s_n2d_7/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/pow_1m
conv_s_n2d_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_7/add/yЃ
conv_s_n2d_7/addAddV2conv_s_n2d_7/pow_1:z:0conv_s_n2d_7/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/addќ
conv_s_n2d_7/truedivRealDivconv_s_n2d_7/MatMul:product:0conv_s_n2d_7/add:z:0*
T0*
_output_shapes
:	└2
conv_s_n2d_7/truedivџ
conv_s_n2d_7/MatMul_1MatMulconv_s_n2d_7/truediv:z:0conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/MatMul_1q
conv_s_n2d_7/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_7/pow_2/yў
conv_s_n2d_7/pow_2Powconv_s_n2d_7/MatMul_1:product:0conv_s_n2d_7/pow_2/y:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/pow_2}
conv_s_n2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/Const_1Є
conv_s_n2d_7/Sum_1Sumconv_s_n2d_7/pow_2:z:0conv_s_n2d_7/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/Sum_1q
conv_s_n2d_7/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_7/pow_3/yї
conv_s_n2d_7/pow_3Powconv_s_n2d_7/Sum_1:output:0conv_s_n2d_7/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/pow_3q
conv_s_n2d_7/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_7/add_1/yЅ
conv_s_n2d_7/add_1AddV2conv_s_n2d_7/pow_3:z:0conv_s_n2d_7/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_7/add_1Ю
conv_s_n2d_7/truediv_1RealDivconv_s_n2d_7/MatMul_1:product:0conv_s_n2d_7/add_1:z:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/truediv_1џ
conv_s_n2d_7/MatMul_2MatMulconv_s_n2d_7/truediv:z:0conv_s_n2d_7/Reshape:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/MatMul_2Ј
conv_s_n2d_7/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_7/transpose_1/perm«
conv_s_n2d_7/transpose_1	Transposeconv_s_n2d_7/truediv_1:z:0&conv_s_n2d_7/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
conv_s_n2d_7/transpose_1а
conv_s_n2d_7/MatMul_3MatMulconv_s_n2d_7/MatMul_2:product:0conv_s_n2d_7/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_7/MatMul_3Ц
conv_s_n2d_7/truediv_2RealDivconv_s_n2d_7/Reshape:output:0conv_s_n2d_7/MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
conv_s_n2d_7/truediv_2с
conv_s_n2d_7/AssignVariableOpAssignVariableOp+conv_s_n2d_7_matmul_readvariableop_resourceconv_s_n2d_7/truediv_1:z:0#^conv_s_n2d_7/MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
conv_s_n2d_7/AssignVariableOpх
conv_s_n2d_7/Reshape_1/shapeConst^conv_s_n2d_7/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      @       2
conv_s_n2d_7/Reshape_1/shape»
conv_s_n2d_7/Reshape_1Reshapeconv_s_n2d_7/truediv_2:z:0%conv_s_n2d_7/Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
conv_s_n2d_7/Reshape_1ж
conv_s_n2d_7/convolutionConv2D&leaky_re_lu_16/LeakyRelu:activations:0conv_s_n2d_7/Reshape_1:output:0*
T0*/
_output_shapes
:         @| *
paddingSAME*
strides
2
conv_s_n2d_7/convolution│
#conv_s_n2d_7/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#conv_s_n2d_7/BiasAdd/ReadVariableOp┴
conv_s_n2d_7/BiasAddBiasAdd!conv_s_n2d_7/convolution:output:0+conv_s_n2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @| 2
conv_s_n2d_7/BiasAdd{
up_sampling2d_2/ShapeShapeconv_s_n2d_7/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shapeћ
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stackў
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1ў
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2«
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Constъ
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mulЃ
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_7/BiasAdd:output:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђЭ *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor│
leaky_re_lu_17/LeakyRelu	LeakyRelu=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ 2
leaky_re_lu_17/LeakyRelu┐
#conv_s_n2d_8/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_8_reshape_readvariableop_resource*&
_output_shapes
: *
dtype02%
#conv_s_n2d_8/Reshape/ReadVariableOpЅ
conv_s_n2d_8/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Reshape/shape│
conv_s_n2d_8/ReshapeReshape+conv_s_n2d_8/Reshape/ReadVariableOp:value:0#conv_s_n2d_8/Reshape/shape:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/ReshapeІ
conv_s_n2d_8/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/transpose/permг
conv_s_n2d_8/transpose	Transposeconv_s_n2d_8/Reshape:output:0$conv_s_n2d_8/transpose/perm:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/transpose┤
"conv_s_n2d_8/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"conv_s_n2d_8/MatMul/ReadVariableOpд
conv_s_n2d_8/MatMulMatMul*conv_s_n2d_8/MatMul/ReadVariableOp:value:0conv_s_n2d_8/transpose:y:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/MatMulm
conv_s_n2d_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_8/pow/yЉ
conv_s_n2d_8/powPowconv_s_n2d_8/MatMul:product:0conv_s_n2d_8/pow/y:output:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/powy
conv_s_n2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Const
conv_s_n2d_8/SumSumconv_s_n2d_8/pow:z:0conv_s_n2d_8/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/Sumq
conv_s_n2d_8/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_8/pow_1/yі
conv_s_n2d_8/pow_1Powconv_s_n2d_8/Sum:output:0conv_s_n2d_8/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/pow_1m
conv_s_n2d_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_8/add/yЃ
conv_s_n2d_8/addAddV2conv_s_n2d_8/pow_1:z:0conv_s_n2d_8/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/addќ
conv_s_n2d_8/truedivRealDivconv_s_n2d_8/MatMul:product:0conv_s_n2d_8/add:z:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/truedivџ
conv_s_n2d_8/MatMul_1MatMulconv_s_n2d_8/truediv:z:0conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_1q
conv_s_n2d_8/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_8/pow_2/yў
conv_s_n2d_8/pow_2Powconv_s_n2d_8/MatMul_1:product:0conv_s_n2d_8/pow_2/y:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/pow_2}
conv_s_n2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/Const_1Є
conv_s_n2d_8/Sum_1Sumconv_s_n2d_8/pow_2:z:0conv_s_n2d_8/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/Sum_1q
conv_s_n2d_8/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_8/pow_3/yї
conv_s_n2d_8/pow_3Powconv_s_n2d_8/Sum_1:output:0conv_s_n2d_8/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/pow_3q
conv_s_n2d_8/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_8/add_1/yЅ
conv_s_n2d_8/add_1AddV2conv_s_n2d_8/pow_3:z:0conv_s_n2d_8/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_8/add_1Ю
conv_s_n2d_8/truediv_1RealDivconv_s_n2d_8/MatMul_1:product:0conv_s_n2d_8/add_1:z:0*
T0*
_output_shapes

:2
conv_s_n2d_8/truediv_1џ
conv_s_n2d_8/MatMul_2MatMulconv_s_n2d_8/truediv:z:0conv_s_n2d_8/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_2Ј
conv_s_n2d_8/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_8/transpose_1/perm«
conv_s_n2d_8/transpose_1	Transposeconv_s_n2d_8/truediv_1:z:0&conv_s_n2d_8/transpose_1/perm:output:0*
T0*
_output_shapes

:2
conv_s_n2d_8/transpose_1а
conv_s_n2d_8/MatMul_3MatMulconv_s_n2d_8/MatMul_2:product:0conv_s_n2d_8/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_8/MatMul_3Ц
conv_s_n2d_8/truediv_2RealDivconv_s_n2d_8/Reshape:output:0conv_s_n2d_8/MatMul_3:product:0*
T0*
_output_shapes
:	а2
conv_s_n2d_8/truediv_2с
conv_s_n2d_8/AssignVariableOpAssignVariableOp+conv_s_n2d_8_matmul_readvariableop_resourceconv_s_n2d_8/truediv_1:z:0#^conv_s_n2d_8/MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
conv_s_n2d_8/AssignVariableOpх
conv_s_n2d_8/Reshape_1/shapeConst^conv_s_n2d_8/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"             2
conv_s_n2d_8/Reshape_1/shape»
conv_s_n2d_8/Reshape_1Reshapeconv_s_n2d_8/truediv_2:z:0%conv_s_n2d_8/Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
conv_s_n2d_8/Reshape_1в
conv_s_n2d_8/convolutionConv2D&leaky_re_lu_17/LeakyRelu:activations:0conv_s_n2d_8/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2
conv_s_n2d_8/convolution│
#conv_s_n2d_8/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#conv_s_n2d_8/BiasAdd/ReadVariableOp├
conv_s_n2d_8/BiasAddBiasAdd!conv_s_n2d_8/convolution:output:0+conv_s_n2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
conv_s_n2d_8/BiasAdd{
up_sampling2d_3/ShapeShapeconv_s_n2d_8/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shapeћ
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stackў
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1ў
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2«
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Constъ
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mulЃ
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_8/BiasAdd:output:0up_sampling2d_3/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor│
leaky_re_lu_18/LeakyRelu	LeakyRelu=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2
leaky_re_lu_18/LeakyRelu┐
#conv_s_n2d_9/Reshape/ReadVariableOpReadVariableOp,conv_s_n2d_9_reshape_readvariableop_resource*&
_output_shapes
:*
dtype02%
#conv_s_n2d_9/Reshape/ReadVariableOpЅ
conv_s_n2d_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Reshape/shape│
conv_s_n2d_9/ReshapeReshape+conv_s_n2d_9/Reshape/ReadVariableOp:value:0#conv_s_n2d_9/Reshape/shape:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/ReshapeІ
conv_s_n2d_9/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/transpose/permг
conv_s_n2d_9/transpose	Transposeconv_s_n2d_9/Reshape:output:0$conv_s_n2d_9/transpose/perm:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/transpose┤
"conv_s_n2d_9/MatMul/ReadVariableOpReadVariableOp+conv_s_n2d_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"conv_s_n2d_9/MatMul/ReadVariableOpд
conv_s_n2d_9/MatMulMatMul*conv_s_n2d_9/MatMul/ReadVariableOp:value:0conv_s_n2d_9/transpose:y:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/MatMulm
conv_s_n2d_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_9/pow/yЉ
conv_s_n2d_9/powPowconv_s_n2d_9/MatMul:product:0conv_s_n2d_9/pow/y:output:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/powy
conv_s_n2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Const
conv_s_n2d_9/SumSumconv_s_n2d_9/pow:z:0conv_s_n2d_9/Const:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/Sumq
conv_s_n2d_9/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_9/pow_1/yі
conv_s_n2d_9/pow_1Powconv_s_n2d_9/Sum:output:0conv_s_n2d_9/pow_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/pow_1m
conv_s_n2d_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_9/add/yЃ
conv_s_n2d_9/addAddV2conv_s_n2d_9/pow_1:z:0conv_s_n2d_9/add/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/addќ
conv_s_n2d_9/truedivRealDivconv_s_n2d_9/MatMul:product:0conv_s_n2d_9/add:z:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/truedivџ
conv_s_n2d_9/MatMul_1MatMulconv_s_n2d_9/truediv:z:0conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_1q
conv_s_n2d_9/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conv_s_n2d_9/pow_2/yў
conv_s_n2d_9/pow_2Powconv_s_n2d_9/MatMul_1:product:0conv_s_n2d_9/pow_2/y:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/pow_2}
conv_s_n2d_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/Const_1Є
conv_s_n2d_9/Sum_1Sumconv_s_n2d_9/pow_2:z:0conv_s_n2d_9/Const_1:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/Sum_1q
conv_s_n2d_9/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_s_n2d_9/pow_3/yї
conv_s_n2d_9/pow_3Powconv_s_n2d_9/Sum_1:output:0conv_s_n2d_9/pow_3/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/pow_3q
conv_s_n2d_9/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
conv_s_n2d_9/add_1/yЅ
conv_s_n2d_9/add_1AddV2conv_s_n2d_9/pow_3:z:0conv_s_n2d_9/add_1/y:output:0*
T0*
_output_shapes
: 2
conv_s_n2d_9/add_1Ю
conv_s_n2d_9/truediv_1RealDivconv_s_n2d_9/MatMul_1:product:0conv_s_n2d_9/add_1:z:0*
T0*
_output_shapes

:2
conv_s_n2d_9/truediv_1џ
conv_s_n2d_9/MatMul_2MatMulconv_s_n2d_9/truediv:z:0conv_s_n2d_9/Reshape:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_2Ј
conv_s_n2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
conv_s_n2d_9/transpose_1/perm«
conv_s_n2d_9/transpose_1	Transposeconv_s_n2d_9/truediv_1:z:0&conv_s_n2d_9/transpose_1/perm:output:0*
T0*
_output_shapes

:2
conv_s_n2d_9/transpose_1а
conv_s_n2d_9/MatMul_3MatMulconv_s_n2d_9/MatMul_2:product:0conv_s_n2d_9/transpose_1:y:0*
T0*
_output_shapes

:2
conv_s_n2d_9/MatMul_3Ц
conv_s_n2d_9/truediv_2RealDivconv_s_n2d_9/Reshape:output:0conv_s_n2d_9/MatMul_3:product:0*
T0*
_output_shapes
:	љ2
conv_s_n2d_9/truediv_2с
conv_s_n2d_9/AssignVariableOpAssignVariableOp+conv_s_n2d_9_matmul_readvariableop_resourceconv_s_n2d_9/truediv_1:z:0#^conv_s_n2d_9/MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
conv_s_n2d_9/AssignVariableOpх
conv_s_n2d_9/Reshape_1/shapeConst^conv_s_n2d_9/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            2
conv_s_n2d_9/Reshape_1/shape»
conv_s_n2d_9/Reshape_1Reshapeconv_s_n2d_9/truediv_2:z:0%conv_s_n2d_9/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
conv_s_n2d_9/Reshape_1в
conv_s_n2d_9/convolutionConv2D&leaky_re_lu_18/LeakyRelu:activations:0conv_s_n2d_9/Reshape_1:output:0*
T0*1
_output_shapes
:         ђЭ*
paddingSAME*
strides
2
conv_s_n2d_9/convolution│
#conv_s_n2d_9/BiasAdd/ReadVariableOpReadVariableOp,conv_s_n2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#conv_s_n2d_9/BiasAdd/ReadVariableOp├
conv_s_n2d_9/BiasAddBiasAdd!conv_s_n2d_9/convolution:output:0+conv_s_n2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђЭ2
conv_s_n2d_9/BiasAdd{
up_sampling2d_4/ShapeShapeconv_s_n2d_9/BiasAdd:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shapeћ
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stackў
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1ў
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2«
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Constъ
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mulЃ
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv_s_n2d_9/BiasAdd:output:0up_sampling2d_4/mul:z:0*
T0*1
_output_shapes
:         ђЭ*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor│
leaky_re_lu_19/LeakyRelu	LeakyRelu=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0*1
_output_shapes
:         ђЭ2
leaky_re_lu_19/LeakyReluй

IdentityIdentity&leaky_re_lu_19/LeakyRelu:activations:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^conv_s_n2d_5/AssignVariableOp$^conv_s_n2d_5/BiasAdd/ReadVariableOp#^conv_s_n2d_5/MatMul/ReadVariableOp$^conv_s_n2d_5/Reshape/ReadVariableOp^conv_s_n2d_6/AssignVariableOp$^conv_s_n2d_6/BiasAdd/ReadVariableOp#^conv_s_n2d_6/MatMul/ReadVariableOp$^conv_s_n2d_6/Reshape/ReadVariableOp^conv_s_n2d_7/AssignVariableOp$^conv_s_n2d_7/BiasAdd/ReadVariableOp#^conv_s_n2d_7/MatMul/ReadVariableOp$^conv_s_n2d_7/Reshape/ReadVariableOp^conv_s_n2d_8/AssignVariableOp$^conv_s_n2d_8/BiasAdd/ReadVariableOp#^conv_s_n2d_8/MatMul/ReadVariableOp$^conv_s_n2d_8/Reshape/ReadVariableOp^conv_s_n2d_9/AssignVariableOp$^conv_s_n2d_9/BiasAdd/ReadVariableOp#^conv_s_n2d_9/MatMul/ReadVariableOp$^conv_s_n2d_9/Reshape/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*1
_output_shapes
:         ђЭ2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2>
conv_s_n2d_5/AssignVariableOpconv_s_n2d_5/AssignVariableOp2J
#conv_s_n2d_5/BiasAdd/ReadVariableOp#conv_s_n2d_5/BiasAdd/ReadVariableOp2H
"conv_s_n2d_5/MatMul/ReadVariableOp"conv_s_n2d_5/MatMul/ReadVariableOp2J
#conv_s_n2d_5/Reshape/ReadVariableOp#conv_s_n2d_5/Reshape/ReadVariableOp2>
conv_s_n2d_6/AssignVariableOpconv_s_n2d_6/AssignVariableOp2J
#conv_s_n2d_6/BiasAdd/ReadVariableOp#conv_s_n2d_6/BiasAdd/ReadVariableOp2H
"conv_s_n2d_6/MatMul/ReadVariableOp"conv_s_n2d_6/MatMul/ReadVariableOp2J
#conv_s_n2d_6/Reshape/ReadVariableOp#conv_s_n2d_6/Reshape/ReadVariableOp2>
conv_s_n2d_7/AssignVariableOpconv_s_n2d_7/AssignVariableOp2J
#conv_s_n2d_7/BiasAdd/ReadVariableOp#conv_s_n2d_7/BiasAdd/ReadVariableOp2H
"conv_s_n2d_7/MatMul/ReadVariableOp"conv_s_n2d_7/MatMul/ReadVariableOp2J
#conv_s_n2d_7/Reshape/ReadVariableOp#conv_s_n2d_7/Reshape/ReadVariableOp2>
conv_s_n2d_8/AssignVariableOpconv_s_n2d_8/AssignVariableOp2J
#conv_s_n2d_8/BiasAdd/ReadVariableOp#conv_s_n2d_8/BiasAdd/ReadVariableOp2H
"conv_s_n2d_8/MatMul/ReadVariableOp"conv_s_n2d_8/MatMul/ReadVariableOp2J
#conv_s_n2d_8/Reshape/ReadVariableOp#conv_s_n2d_8/Reshape/ReadVariableOp2>
conv_s_n2d_9/AssignVariableOpconv_s_n2d_9/AssignVariableOp2J
#conv_s_n2d_9/BiasAdd/ReadVariableOp#conv_s_n2d_9/BiasAdd/ReadVariableOp2H
"conv_s_n2d_9/MatMul/ReadVariableOp"conv_s_n2d_9/MatMul/ReadVariableOp2J
#conv_s_n2d_9/Reshape/ReadVariableOp#conv_s_n2d_9/Reshape/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs/1
Ѕ
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1472501

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
з
њ
.__inference_conv_s_n2d_7_layer_call_fn_1475549

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726402
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ж
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1475090

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:          >2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы
Ъ
D__inference_dense_1_layer_call_and_return_conditional_losses_1474862

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0**
_input_shapes
:         :2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
с
)__inference_model_1_layer_call_fn_1474069
input_3
input_4
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_14740182
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
╦'
│
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1472425

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	ђ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	ђ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	ђ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:@2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:@2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:@2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_1475565

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                            2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
К'
│
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475613

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	а2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	а2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	а2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	а2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	а2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	а2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┬%
а
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475538

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	 └2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	└2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	└2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	└2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

: 2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

: 2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
─
g
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_1473213

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         @2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
├
o
)__inference_dense_5_layer_call_fn_1474931

inputs
unknown
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_14732602
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_1475447

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╚
g
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_1475042

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         └2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ј
L
0__inference_leaky_re_lu_16_layer_call_fn_1475452

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_14735762
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┬%
а
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475774

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	љ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	љ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	љ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	љ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	љ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	љ2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш&
│
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475151

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Kђ2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	ђK2
	transposeј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOpq
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes

:K2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes

:K2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
adda
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes

:K2	
truedivg
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/ye
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes
:	ђ2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes
:	ђ2
	truediv_1g
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes
:	ђ2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes
:	ђ2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	Kђ2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"         ђ   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ2
	Reshape_1Б
convolutionConv2DinputsReshape_1:output:0*
T0*0
_output_shapes
:          >ђ*
paddingSAME*
strides
2
convolutionЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpј
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          >ђ2	
BiasAdd╩
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*0
_output_shapes
:          >ђ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':          >:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:W S
/
_output_shapes
:          >
 
_user_specified_nameinputs
Ѕ
h
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1473146

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2╬
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulН
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(2
resize/ResizeNearestNeighborц
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г0
╔
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1472033

inputs
assignmovingavg_1472008
assignmovingavg_1_1472014)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	└2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         └2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	└*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:└*
squeeze_dims
 2
moments/Squeeze_1═
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/1472008*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
AssignMovingAvg/decayЋ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1472008*
_output_shapes	
:└*
dtype02 
AssignMovingAvg/ReadVariableOpз
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/1472008*
_output_shapes	
:└2
AssignMovingAvg/subЖ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg/1472008*
_output_shapes	
:└2
AssignMovingAvg/mul▒
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1472008AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg/1472008*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpМ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/1472014*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
AssignMovingAvg_1/decayЏ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1472014*
_output_shapes	
:└*
dtype02"
 AssignMovingAvg_1/ReadVariableOp§
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1472014*
_output_shapes	
:└2
AssignMovingAvg_1/subЗ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1472014*
_output_shapes	
:└2
AssignMovingAvg_1/mulй
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1472014AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*,
_class"
 loc:@AssignMovingAvg_1/1472014*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
З
њ
.__inference_conv_s_n2d_8_layer_call_fn_1475678

inputs
unknown
	unknown_0
	unknown_1
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14729092
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
З
Ъ
D__inference_dense_5_layer_call_and_return_conditional_losses_1474924

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d└*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
јz
 
D__inference_model_1_layer_call_and_return_conditional_losses_1473714
input_3
input_4
dense_1_1473173
dense_2_1473205
dense_3_1473237
dense_5_1473269
dense_4_1473288
batch_normalization_1473317
batch_normalization_1473319
batch_normalization_1473321
batch_normalization_1473323
conv_s_n2d_5_1473520
conv_s_n2d_5_1473522
conv_s_n2d_5_1473524
conv_s_n2d_6_1473563
conv_s_n2d_6_1473565
conv_s_n2d_6_1473567
conv_s_n2d_7_1473606
conv_s_n2d_7_1473608
conv_s_n2d_7_1473610
conv_s_n2d_8_1473649
conv_s_n2d_8_1473651
conv_s_n2d_8_1473653
conv_s_n2d_9_1473692
conv_s_n2d_9_1473694
conv_s_n2d_9_1473696
identityѕб+batch_normalization/StatefulPartitionedCallб$conv_s_n2d_5/StatefulPartitionedCallб$conv_s_n2d_6/StatefulPartitionedCallб$conv_s_n2d_7/StatefulPartitionedCallб$conv_s_n2d_8/StatefulPartitionedCallб$conv_s_n2d_9/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallЁ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_1_1473173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14731642!
dense_1/StatefulPartitionedCallј
leaky_re_lu_10/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_14731812 
leaky_re_lu_10/PartitionedCallЦ
dense_2/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0dense_2_1473205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_14731962!
dense_2/StatefulPartitionedCallј
leaky_re_lu_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_14732132 
leaky_re_lu_11/PartitionedCallд
dense_3/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0dense_3_1473237*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_14732282!
dense_3/StatefulPartitionedCallЈ
leaky_re_lu_12/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_14732452 
leaky_re_lu_12/PartitionedCallє
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_5_1473269*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_14732602!
dense_5/StatefulPartitionedCallд
dense_4/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0dense_4_1473288*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_14732792!
dense_4/StatefulPartitionedCall▓
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_1473317batch_normalization_1473319batch_normalization_1473321batch_normalization_1473323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720332-
+batch_normalization/StatefulPartitionedCallЈ
leaky_re_lu_13/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_14733312 
leaky_re_lu_13/PartitionedCallЏ
leaky_re_lu_14/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_14733442 
leaky_re_lu_14/PartitionedCallє
reshape_2/PartitionedCallPartitionedCall'leaky_re_lu_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_14733662
reshape_2/PartitionedCallє
reshape_1/PartitionedCallPartitionedCall'leaky_re_lu_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_14733882
reshape_1/PartitionedCall▓
concatenate_2/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_14734032
concatenate_2/PartitionedCall­
$conv_s_n2d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_s_n2d_5_1473520conv_s_n2d_5_1473522conv_s_n2d_5_1473524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734542&
$conv_s_n2d_5/StatefulPartitionedCallФ
up_sampling2d/PartitionedCallPartitionedCall-conv_s_n2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_14722862
up_sampling2d/PartitionedCallД
leaky_re_lu_15/PartitionedCallPartitionedCall&up_sampling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_14735332 
leaky_re_lu_15/PartitionedCallѓ
$conv_s_n2d_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_15/PartitionedCall:output:0conv_s_n2d_6_1473563conv_s_n2d_6_1473565conv_s_n2d_6_1473567*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724252&
$conv_s_n2d_6/StatefulPartitionedCall░
up_sampling2d_1/PartitionedCallPartitionedCall-conv_s_n2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_14725012!
up_sampling2d_1/PartitionedCallе
leaky_re_lu_16/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_14735762 
leaky_re_lu_16/PartitionedCallѓ
$conv_s_n2d_7/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0conv_s_n2d_7_1473606conv_s_n2d_7_1473608conv_s_n2d_7_1473610*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726402&
$conv_s_n2d_7/StatefulPartitionedCall░
up_sampling2d_2/PartitionedCallPartitionedCall-conv_s_n2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_14727162!
up_sampling2d_2/PartitionedCallе
leaky_re_lu_17/PartitionedCallPartitionedCall(up_sampling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_14736192 
leaky_re_lu_17/PartitionedCallѓ
$conv_s_n2d_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv_s_n2d_8_1473649conv_s_n2d_8_1473651conv_s_n2d_8_1473653*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14728552&
$conv_s_n2d_8/StatefulPartitionedCall░
up_sampling2d_3/PartitionedCallPartitionedCall-conv_s_n2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14729312!
up_sampling2d_3/PartitionedCallе
leaky_re_lu_18/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_14736622 
leaky_re_lu_18/PartitionedCallѓ
$conv_s_n2d_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0conv_s_n2d_9_1473692conv_s_n2d_9_1473694conv_s_n2d_9_1473696*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14730702&
$conv_s_n2d_9/StatefulPartitionedCall░
up_sampling2d_4/PartitionedCallPartitionedCall-conv_s_n2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_14731462!
up_sampling2d_4/PartitionedCallе
leaky_re_lu_19/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_14737052 
leaky_re_lu_19/PartitionedCall░
IdentityIdentity'leaky_re_lu_19/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^conv_s_n2d_5/StatefulPartitionedCall%^conv_s_n2d_6/StatefulPartitionedCall%^conv_s_n2d_7/StatefulPartitionedCall%^conv_s_n2d_8/StatefulPartitionedCall%^conv_s_n2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$conv_s_n2d_5/StatefulPartitionedCall$conv_s_n2d_5/StatefulPartitionedCall2L
$conv_s_n2d_6/StatefulPartitionedCall$conv_s_n2d_6/StatefulPartitionedCall2L
$conv_s_n2d_7/StatefulPartitionedCall$conv_s_n2d_7/StatefulPartitionedCall2L
$conv_s_n2d_8/StatefulPartitionedCall$conv_s_n2d_8/StatefulPartitionedCall2L
$conv_s_n2d_9/StatefulPartitionedCall$conv_s_n2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
╚
g
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_1475052

inputs
identityU
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђ2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ж
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1473366

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:          >2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
к%
а
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475420

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	ђ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	ђ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	ђ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:@2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:@2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:@2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Чe
Ж
#__inference__traced_restore_1475984
file_prefix#
assignvariableop_dense_1_kernel%
!assignvariableop_1_dense_2_kernel%
!assignvariableop_2_dense_3_kernel%
!assignvariableop_3_dense_5_kernel0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance%
!assignvariableop_8_dense_4_kernel*
&assignvariableop_9_conv_s_n2d_5_kernel)
%assignvariableop_10_conv_s_n2d_5_bias'
#assignvariableop_11_conv_s_n2d_5_sn+
'assignvariableop_12_conv_s_n2d_6_kernel)
%assignvariableop_13_conv_s_n2d_6_bias'
#assignvariableop_14_conv_s_n2d_6_sn+
'assignvariableop_15_conv_s_n2d_7_kernel)
%assignvariableop_16_conv_s_n2d_7_bias'
#assignvariableop_17_conv_s_n2d_7_sn+
'assignvariableop_18_conv_s_n2d_8_kernel)
%assignvariableop_19_conv_s_n2d_8_bias'
#assignvariableop_20_conv_s_n2d_8_sn+
'assignvariableop_21_conv_s_n2d_9_kernel)
%assignvariableop_22_conv_s_n2d_9_bias'
#assignvariableop_23_conv_s_n2d_9_sn
identity_25ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9О
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с

value┘
Bо
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-6/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-7/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-8/sn/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-9/sn/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-10/sn/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names└
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesе
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityъ
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2д
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_5_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5░
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6и
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╗
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ф
AssignVariableOp_9AssignVariableOp&assignvariableop_9_conv_s_n2d_5_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Г
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv_s_n2d_5_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ф
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv_s_n2d_5_snIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12»
AssignVariableOp_12AssignVariableOp'assignvariableop_12_conv_s_n2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Г
AssignVariableOp_13AssignVariableOp%assignvariableop_13_conv_s_n2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ф
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv_s_n2d_6_snIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15»
AssignVariableOp_15AssignVariableOp'assignvariableop_15_conv_s_n2d_7_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Г
AssignVariableOp_16AssignVariableOp%assignvariableop_16_conv_s_n2d_7_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ф
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv_s_n2d_7_snIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18»
AssignVariableOp_18AssignVariableOp'assignvariableop_18_conv_s_n2d_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Г
AssignVariableOp_19AssignVariableOp%assignvariableop_19_conv_s_n2d_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ф
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv_s_n2d_8_snIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21»
AssignVariableOp_21AssignVariableOp'assignvariableop_21_conv_s_n2d_9_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Г
AssignVariableOp_22AssignVariableOp%assignvariableop_22_conv_s_n2d_9_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ф
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv_s_n2d_9_snIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЬ
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24р
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
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
AssignVariableOp_23AssignVariableOp_232(
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
╦'
│
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475377

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбAssignVariableOpбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpЎ
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:ђ@*
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ@2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	@ђ2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	ђ2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	ђ2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	ђ2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:@2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:@2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:@2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:@2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	ђ@2
	truediv_2б
AssignVariableOpAssignVariableOpmatmul_readvariableop_resourcetruediv_1:z:0^MatMul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpј
Reshape_1/shapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"      ђ   @   2
Reshape_1/shape|
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ђ@2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAdd█
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
їz
 
D__inference_model_1_layer_call_and_return_conditional_losses_1473882

inputs
inputs_1
dense_1_1473804
dense_2_1473808
dense_3_1473812
dense_5_1473816
dense_4_1473819
batch_normalization_1473822
batch_normalization_1473824
batch_normalization_1473826
batch_normalization_1473828
conv_s_n2d_5_1473836
conv_s_n2d_5_1473838
conv_s_n2d_5_1473840
conv_s_n2d_6_1473845
conv_s_n2d_6_1473847
conv_s_n2d_6_1473849
conv_s_n2d_7_1473854
conv_s_n2d_7_1473856
conv_s_n2d_7_1473858
conv_s_n2d_8_1473863
conv_s_n2d_8_1473865
conv_s_n2d_8_1473867
conv_s_n2d_9_1473872
conv_s_n2d_9_1473874
conv_s_n2d_9_1473876
identityѕб+batch_normalization/StatefulPartitionedCallб$conv_s_n2d_5/StatefulPartitionedCallб$conv_s_n2d_6/StatefulPartitionedCallб$conv_s_n2d_7/StatefulPartitionedCallб$conv_s_n2d_8/StatefulPartitionedCallб$conv_s_n2d_9/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallё
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_1473804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14731642!
dense_1/StatefulPartitionedCallј
leaky_re_lu_10/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_14731812 
leaky_re_lu_10/PartitionedCallЦ
dense_2/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0dense_2_1473808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_14731962!
dense_2/StatefulPartitionedCallј
leaky_re_lu_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_14732132 
leaky_re_lu_11/PartitionedCallд
dense_3/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0dense_3_1473812*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_14732282!
dense_3/StatefulPartitionedCallЈ
leaky_re_lu_12/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_14732452 
leaky_re_lu_12/PartitionedCallЄ
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_5_1473816*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_14732602!
dense_5/StatefulPartitionedCallд
dense_4/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0dense_4_1473819*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_14732792!
dense_4/StatefulPartitionedCall▓
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_1473822batch_normalization_1473824batch_normalization_1473826batch_normalization_1473828*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720332-
+batch_normalization/StatefulPartitionedCallЈ
leaky_re_lu_13/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_14733312 
leaky_re_lu_13/PartitionedCallЏ
leaky_re_lu_14/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_14733442 
leaky_re_lu_14/PartitionedCallє
reshape_2/PartitionedCallPartitionedCall'leaky_re_lu_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_14733662
reshape_2/PartitionedCallє
reshape_1/PartitionedCallPartitionedCall'leaky_re_lu_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_14733882
reshape_1/PartitionedCall▓
concatenate_2/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_14734032
concatenate_2/PartitionedCall­
$conv_s_n2d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_s_n2d_5_1473836conv_s_n2d_5_1473838conv_s_n2d_5_1473840*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734542&
$conv_s_n2d_5/StatefulPartitionedCallФ
up_sampling2d/PartitionedCallPartitionedCall-conv_s_n2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_14722862
up_sampling2d/PartitionedCallД
leaky_re_lu_15/PartitionedCallPartitionedCall&up_sampling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_14735332 
leaky_re_lu_15/PartitionedCallѓ
$conv_s_n2d_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_15/PartitionedCall:output:0conv_s_n2d_6_1473845conv_s_n2d_6_1473847conv_s_n2d_6_1473849*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724252&
$conv_s_n2d_6/StatefulPartitionedCall░
up_sampling2d_1/PartitionedCallPartitionedCall-conv_s_n2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_14725012!
up_sampling2d_1/PartitionedCallе
leaky_re_lu_16/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_14735762 
leaky_re_lu_16/PartitionedCallѓ
$conv_s_n2d_7/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0conv_s_n2d_7_1473854conv_s_n2d_7_1473856conv_s_n2d_7_1473858*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726402&
$conv_s_n2d_7/StatefulPartitionedCall░
up_sampling2d_2/PartitionedCallPartitionedCall-conv_s_n2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_14727162!
up_sampling2d_2/PartitionedCallе
leaky_re_lu_17/PartitionedCallPartitionedCall(up_sampling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_14736192 
leaky_re_lu_17/PartitionedCallѓ
$conv_s_n2d_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv_s_n2d_8_1473863conv_s_n2d_8_1473865conv_s_n2d_8_1473867*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14728552&
$conv_s_n2d_8/StatefulPartitionedCall░
up_sampling2d_3/PartitionedCallPartitionedCall-conv_s_n2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14729312!
up_sampling2d_3/PartitionedCallе
leaky_re_lu_18/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_14736622 
leaky_re_lu_18/PartitionedCallѓ
$conv_s_n2d_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0conv_s_n2d_9_1473872conv_s_n2d_9_1473874conv_s_n2d_9_1473876*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14730702&
$conv_s_n2d_9/StatefulPartitionedCall░
up_sampling2d_4/PartitionedCallPartitionedCall-conv_s_n2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_14731462!
up_sampling2d_4/PartitionedCallе
leaky_re_lu_19/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_14737052 
leaky_re_lu_19/PartitionedCall░
IdentityIdentity'leaky_re_lu_19/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^conv_s_n2d_5/StatefulPartitionedCall%^conv_s_n2d_6/StatefulPartitionedCall%^conv_s_n2d_7/StatefulPartitionedCall%^conv_s_n2d_8/StatefulPartitionedCall%^conv_s_n2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$conv_s_n2d_5/StatefulPartitionedCall$conv_s_n2d_5/StatefulPartitionedCall2L
$conv_s_n2d_6/StatefulPartitionedCall$conv_s_n2d_6/StatefulPartitionedCall2L
$conv_s_n2d_7/StatefulPartitionedCall$conv_s_n2d_7/StatefulPartitionedCall2L
$conv_s_n2d_8/StatefulPartitionedCall$conv_s_n2d_8/StatefulPartitionedCall2L
$conv_s_n2d_9/StatefulPartitionedCall$conv_s_n2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         d
 
_user_specified_nameinputs
ы
т
)__inference_model_1_layer_call_fn_1474801
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_14738822
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         d
"
_user_specified_name
inputs/1
┬%
а
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1472694

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	└ 2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	 └2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	└2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	└2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	└2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

: 2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

: 2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

: 2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	└ 2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
:@ 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_1473576

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┬%
а
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1472909

inputs#
reshape_readvariableop_resource"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбReshape/ReadVariableOpў
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype02
Reshape/ReadVariableOpo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	а2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permx
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	а2
	transposeЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpr
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*
_output_shapes
:	а2
MatMulS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y]
powPowMatMul:product:0pow/y:output:0*
T0*
_output_shapes
:	а2
pow_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
SumSumpow:z:0Const:output:0*
T0*
_output_shapes
: 2
SumW
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_1/yV
pow_1PowSum:output:0pow_1/y:output:0*
T0*
_output_shapes
: 2
pow_1S
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2
add/yO
addAddV2	pow_1:z:0add/y:output:0*
T0*
_output_shapes
: 2
addb
truedivRealDivMatMul:product:0add:z:0*
T0*
_output_shapes
:	а2	
truedivf
MatMul_1MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_1W
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_2/yd
pow_2PowMatMul_1:product:0pow_2/y:output:0*
T0*
_output_shapes

:2
pow_2c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1S
Sum_1Sum	pow_2:z:0Const_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
pow_3/yX
pow_3PowSum_1:output:0pow_3/y:output:0*
T0*
_output_shapes
: 2
pow_3W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+2	
add_1/yU
add_1AddV2	pow_3:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1i
	truediv_1RealDivMatMul_1:product:0	add_1:z:0*
T0*
_output_shapes

:2
	truediv_1f
MatMul_2MatMultruediv:z:0Reshape:output:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetruediv_1:z:0transpose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1l
MatMul_3MatMulMatMul_2:product:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_3q
	truediv_2RealDivReshape:output:0MatMul_3:product:0*
T0*
_output_shapes
:	а2
	truediv_2{
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
Reshape_1/shape{
	Reshape_1Reshapetruediv_2:z:0Reshape_1/shape:output:0*
T0*&
_output_shapes
: 2
	Reshape_1┤
convolutionConv2DinputsReshape_1:output:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
convolutionї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd╚
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Reshape/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┤
M
1__inference_up_sampling2d_2_layer_call_fn_1472722

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_14727162
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ж
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1473388

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:          >2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:          >2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
├
o
)__inference_dense_3_layer_call_fn_1474917

inputs
unknown
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_14732282
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0**
_input_shapes
:         @:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
э
Ъ
D__inference_dense_4_layer_call_and_return_conditional_losses_1473279

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_1475801

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
g
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_1473662

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┼
o
)__inference_dense_4_layer_call_fn_1475037

inputs
unknown
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_14732792
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ћz
 
D__inference_model_1_layer_call_and_return_conditional_losses_1473796
input_3
input_4
dense_1_1473718
dense_2_1473722
dense_3_1473726
dense_5_1473730
dense_4_1473733
batch_normalization_1473736
batch_normalization_1473738
batch_normalization_1473740
batch_normalization_1473742
conv_s_n2d_5_1473750
conv_s_n2d_5_1473752
conv_s_n2d_5_1473754
conv_s_n2d_6_1473759
conv_s_n2d_6_1473761
conv_s_n2d_6_1473763
conv_s_n2d_7_1473768
conv_s_n2d_7_1473770
conv_s_n2d_7_1473772
conv_s_n2d_8_1473777
conv_s_n2d_8_1473779
conv_s_n2d_8_1473781
conv_s_n2d_9_1473786
conv_s_n2d_9_1473788
conv_s_n2d_9_1473790
identityѕб+batch_normalization/StatefulPartitionedCallб$conv_s_n2d_5/StatefulPartitionedCallб$conv_s_n2d_6/StatefulPartitionedCallб$conv_s_n2d_7/StatefulPartitionedCallб$conv_s_n2d_8/StatefulPartitionedCallб$conv_s_n2d_9/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallЁ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_1_1473718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14731642!
dense_1/StatefulPartitionedCallј
leaky_re_lu_10/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_14731812 
leaky_re_lu_10/PartitionedCallЦ
dense_2/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0dense_2_1473722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_14731962!
dense_2/StatefulPartitionedCallј
leaky_re_lu_11/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_14732132 
leaky_re_lu_11/PartitionedCallд
dense_3/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0dense_3_1473726*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_14732282!
dense_3/StatefulPartitionedCallЈ
leaky_re_lu_12/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_14732452 
leaky_re_lu_12/PartitionedCallє
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_5_1473730*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_14732602!
dense_5/StatefulPartitionedCallд
dense_4/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_12/PartitionedCall:output:0dense_4_1473733*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_14732792!
dense_4/StatefulPartitionedCall┤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_1473736batch_normalization_1473738batch_normalization_1473740batch_normalization_1473742*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14720662-
+batch_normalization/StatefulPartitionedCallЈ
leaky_re_lu_13/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_14733312 
leaky_re_lu_13/PartitionedCallЏ
leaky_re_lu_14/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_14733442 
leaky_re_lu_14/PartitionedCallє
reshape_2/PartitionedCallPartitionedCall'leaky_re_lu_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_14733662
reshape_2/PartitionedCallє
reshape_1/PartitionedCallPartitionedCall'leaky_re_lu_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_14733882
reshape_1/PartitionedCall▓
concatenate_2/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          >* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_14734032
concatenate_2/PartitionedCallы
$conv_s_n2d_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv_s_n2d_5_1473750conv_s_n2d_5_1473752conv_s_n2d_5_1473754*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          >ђ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_14734972&
$conv_s_n2d_5/StatefulPartitionedCallФ
up_sampling2d/PartitionedCallPartitionedCall-conv_s_n2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_14722862
up_sampling2d/PartitionedCallД
leaky_re_lu_15/PartitionedCallPartitionedCall&up_sampling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_14735332 
leaky_re_lu_15/PartitionedCallЃ
$conv_s_n2d_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_15/PartitionedCall:output:0conv_s_n2d_6_1473759conv_s_n2d_6_1473761conv_s_n2d_6_1473763*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_14724792&
$conv_s_n2d_6/StatefulPartitionedCall░
up_sampling2d_1/PartitionedCallPartitionedCall-conv_s_n2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_14725012!
up_sampling2d_1/PartitionedCallе
leaky_re_lu_16/PartitionedCallPartitionedCall(up_sampling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_14735762 
leaky_re_lu_16/PartitionedCallЃ
$conv_s_n2d_7/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0conv_s_n2d_7_1473768conv_s_n2d_7_1473770conv_s_n2d_7_1473772*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_14726942&
$conv_s_n2d_7/StatefulPartitionedCall░
up_sampling2d_2/PartitionedCallPartitionedCall-conv_s_n2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_14727162!
up_sampling2d_2/PartitionedCallе
leaky_re_lu_17/PartitionedCallPartitionedCall(up_sampling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_14736192 
leaky_re_lu_17/PartitionedCallЃ
$conv_s_n2d_8/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0conv_s_n2d_8_1473777conv_s_n2d_8_1473779conv_s_n2d_8_1473781*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_14729092&
$conv_s_n2d_8/StatefulPartitionedCall░
up_sampling2d_3/PartitionedCallPartitionedCall-conv_s_n2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_14729312!
up_sampling2d_3/PartitionedCallе
leaky_re_lu_18/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_14736622 
leaky_re_lu_18/PartitionedCallЃ
$conv_s_n2d_9/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0conv_s_n2d_9_1473786conv_s_n2d_9_1473788conv_s_n2d_9_1473790*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *R
fMRK
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_14731242&
$conv_s_n2d_9/StatefulPartitionedCall░
up_sampling2d_4/PartitionedCallPartitionedCall-conv_s_n2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_14731462!
up_sampling2d_4/PartitionedCallе
leaky_re_lu_19/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *T
fORM
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_14737052 
leaky_re_lu_19/PartitionedCall░
IdentityIdentity'leaky_re_lu_19/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^conv_s_n2d_5/StatefulPartitionedCall%^conv_s_n2d_6/StatefulPartitionedCall%^conv_s_n2d_7/StatefulPartitionedCall%^conv_s_n2d_8/StatefulPartitionedCall%^conv_s_n2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*Џ
_input_shapesЅ
є:         :         d::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$conv_s_n2d_5/StatefulPartitionedCall$conv_s_n2d_5/StatefulPartitionedCall2L
$conv_s_n2d_6/StatefulPartitionedCall$conv_s_n2d_6/StatefulPartitionedCall2L
$conv_s_n2d_7/StatefulPartitionedCall$conv_s_n2d_7/StatefulPartitionedCall2L
$conv_s_n2d_8/StatefulPartitionedCall$conv_s_n2d_8/StatefulPartitionedCall2L
$conv_s_n2d_9/StatefulPartitionedCall$conv_s_n2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3:PL
'
_output_shapes
:         d
!
_user_specified_name	input_4
р
Є
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474997

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:└2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:└2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:└*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:└2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         └2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:└2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:└*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:└2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         └2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         └::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Э
serving_defaultС
;
input_30
serving_default_input_3:0         
;
input_40
serving_default_input_4:0         dL
leaky_re_lu_19:
StatefulPartitionedCall:0         ђЭtensorflow/serving/predict:╝Ъ
Ы╔
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer-18
layer_with_weights-7
layer-19
layer-20
layer-21
layer_with_weights-8
layer-22
layer-23
layer-24
layer_with_weights-9
layer-25
layer-26
layer-27
layer_with_weights-10
layer-28
layer-29
layer-30
 trainable_variables
!	variables
"regularization_losses
#	keras_api
$
signatures
╚_default_save_signature
╔__call__
+╩&call_and_return_all_conditional_losses"в┬
_tf_keras_network╬┬{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1984, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_12", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_12", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["leaky_re_lu_12", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_14", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_14", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_13", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_13", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 1]}}, "name": "reshape_2", "inbound_nodes": [[["leaky_re_lu_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 2]}}, "name": "reshape_1", "inbound_nodes": [[["leaky_re_lu_13", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["reshape_2", 0, 0, {}], ["reshape_1", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_5", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv_s_n2d_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_15", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_15", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_6", "inbound_nodes": [[["leaky_re_lu_15", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv_s_n2d_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_16", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_16", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_7", "inbound_nodes": [[["leaky_re_lu_16", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv_s_n2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_17", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_17", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_8", "inbound_nodes": [[["leaky_re_lu_17", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["conv_s_n2d_8", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_18", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_18", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_9", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_9", "inbound_nodes": [[["leaky_re_lu_18", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["conv_s_n2d_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_19", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_19", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["leaky_re_lu_19", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3]}, {"class_name": "TensorShape", "items": [null, 100]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1984, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_12", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_12", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["leaky_re_lu_12", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_14", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_14", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_13", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_13", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 1]}}, "name": "reshape_2", "inbound_nodes": [[["leaky_re_lu_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 2]}}, "name": "reshape_1", "inbound_nodes": [[["leaky_re_lu_13", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["reshape_2", 0, 0, {}], ["reshape_1", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_5", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv_s_n2d_5", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_15", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_15", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_6", "inbound_nodes": [[["leaky_re_lu_15", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv_s_n2d_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_16", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_16", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_7", "inbound_nodes": [[["leaky_re_lu_16", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv_s_n2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_17", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_17", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_8", "inbound_nodes": [[["leaky_re_lu_17", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["conv_s_n2d_8", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_18", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_18", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "ConvSN2D", "config": {"name": "conv_s_n2d_9", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_s_n2d_9", "inbound_nodes": [[["leaky_re_lu_18", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["conv_s_n2d_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_19", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_19", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["leaky_re_lu_19", 0, 0]]}}}
ж"Т
_tf_keras_input_layerк{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
Ё

%kernel
&trainable_variables
'	variables
(regularization_losses
)	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"У
_tf_keras_layer╬{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
Р
*trainable_variables
+	variables
,regularization_losses
-	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Є

.kernel
/trainable_variables
0	variables
1regularization_losses
2	keras_api
¤__call__
+л&call_and_return_all_conditional_losses"Ж
_tf_keras_layerл{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Р
3trainable_variables
4	variables
5regularization_losses
6	keras_api
Л__call__
+м&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ь"Ж
_tf_keras_input_layer╩{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ѕ

7kernel
8trainable_variables
9	variables
:regularization_losses
;	keras_api
М__call__
+н&call_and_return_all_conditional_losses"в
_tf_keras_layerЛ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
І

<kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api
Н__call__
+о&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1984, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Р
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
О__call__
+п&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_12", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│	
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"П
_tf_keras_layer├{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.8, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1984}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1984]}}
І

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Р
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
П__call__
+я&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_14", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Р
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_13", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ч
[trainable_variables
\	variables
]regularization_losses
^	keras_api
р__call__
+Р&call_and_return_all_conditional_losses"Ж
_tf_keras_layerл{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 1]}}}
ч
_trainable_variables
`	variables
aregularization_losses
b	keras_api
с__call__
+С&call_and_return_all_conditional_losses"Ж
_tf_keras_layerл{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 62, 2]}}}
П
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
т__call__
+Т&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 62, 1]}, {"class_name": "TensorShape", "items": [null, 32, 62, 2]}]}
Е


gkernel
hbias
isn
iu
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
у__call__
+У&call_and_return_all_conditional_losses"з
_tf_keras_layer┘{"class_name": "ConvSN2D", "name": "conv_s_n2d_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_s_n2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 62, 3]}}
К
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
ж__call__
+Ж&call_and_return_all_conditional_losses"Х
_tf_keras_layerю{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Р
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
в__call__
+В&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_15", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
г


vkernel
wbias
xsn
xu
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses"Ш
_tf_keras_layer▄{"class_name": "ConvSN2D", "name": "conv_s_n2d_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_s_n2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 62, 128]}}
╠
}trainable_variables
~	variables
regularization_losses
ђ	keras_api
№__call__
+­&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Т
Ђtrainable_variables
ѓ	variables
Ѓregularization_losses
ё	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_16", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│

Ёkernel
	єbias
Єsn
Єu
ѕtrainable_variables
Ѕ	variables
іregularization_losses
І	keras_api
з__call__
+З&call_and_return_all_conditional_losses"ш
_tf_keras_layer█{"class_name": "ConvSN2D", "name": "conv_s_n2d_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_s_n2d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 124, 64]}}
¤
їtrainable_variables
Ї	variables
јregularization_losses
Ј	keras_api
ш__call__
+Ш&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Т
љtrainable_variables
Љ	variables
њregularization_losses
Њ	keras_api
э__call__
+Э&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_17", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
┤

ћkernel
	Ћbias
ќsn
ќu
Ќtrainable_variables
ў	variables
Ўregularization_losses
џ	keras_api
щ__call__
+Щ&call_and_return_all_conditional_losses"Ш
_tf_keras_layer▄{"class_name": "ConvSN2D", "name": "conv_s_n2d_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_s_n2d_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 248, 32]}}
¤
Џtrainable_variables
ю	variables
Юregularization_losses
ъ	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Т
Ъtrainable_variables
а	variables
Аregularization_losses
б	keras_api
§__call__
+■&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_18", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
│

Бkernel
	цbias
Цsn
Цu
дtrainable_variables
Д	variables
еregularization_losses
Е	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses"ш
_tf_keras_layer█{"class_name": "ConvSN2D", "name": "conv_s_n2d_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_s_n2d_9", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 248, 16]}}
¤
фtrainable_variables
Ф	variables
гregularization_losses
Г	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Т
«trainable_variables
»	variables
░regularization_losses
▒	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_19", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ц
%0
.1
72
<3
F4
G5
N6
g7
h8
v9
w10
Ё11
є12
ћ13
Ћ14
Б15
ц16"
trackable_list_wrapper
▀
%0
.1
72
<3
F4
G5
H6
I7
N8
g9
h10
i11
v12
w13
x14
Ё15
є16
Є17
ћ18
Ћ19
ќ20
Б21
ц22
Ц23"
trackable_list_wrapper
 "
trackable_list_wrapper
М
 trainable_variables
!	variables
▓layers
│metrics
┤layer_metrics
 хlayer_regularization_losses
"regularization_losses
Хnon_trainable_variables
╔__call__
╚_default_save_signature
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
-
Ёserving_default"
signature_map
 : 2dense_1/kernel
'
%0"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
х
&trainable_variables
'	variables
иlayers
Иmetrics
╣layer_metrics
 ║layer_regularization_losses
(regularization_losses
╗non_trainable_variables
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
*trainable_variables
+	variables
╝layers
йmetrics
Йlayer_metrics
 ┐layer_regularization_losses
,regularization_losses
└non_trainable_variables
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 : @2dense_2/kernel
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
х
/trainable_variables
0	variables
┴layers
┬metrics
├layer_metrics
 ─layer_regularization_losses
1regularization_losses
┼non_trainable_variables
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
3trainable_variables
4	variables
кlayers
Кmetrics
╚layer_metrics
 ╔layer_regularization_losses
5regularization_losses
╩non_trainable_variables
Л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
!:	@ђ2dense_3/kernel
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
х
8trainable_variables
9	variables
╦layers
╠metrics
═layer_metrics
 ╬layer_regularization_losses
:regularization_losses
¤non_trainable_variables
М__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
!:	d└2dense_5/kernel
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
х
=trainable_variables
>	variables
лlayers
Лmetrics
мlayer_metrics
 Мlayer_regularization_losses
?regularization_losses
нnon_trainable_variables
Н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Atrainable_variables
B	variables
Нlayers
оmetrics
Оlayer_metrics
 пlayer_regularization_losses
Cregularization_losses
┘non_trainable_variables
О__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&└2batch_normalization/gamma
':%└2batch_normalization/beta
0:.└ (2batch_normalization/moving_mean
4:2└ (2#batch_normalization/moving_variance
.
F0
G1"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Jtrainable_variables
K	variables
┌layers
█metrics
▄layer_metrics
 Пlayer_regularization_losses
Lregularization_losses
яnon_trainable_variables
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
": 
ђђ2dense_4/kernel
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Otrainable_variables
P	variables
▀layers
Яmetrics
рlayer_metrics
 Рlayer_regularization_losses
Qregularization_losses
сnon_trainable_variables
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Strainable_variables
T	variables
Сlayers
тmetrics
Тlayer_metrics
 уlayer_regularization_losses
Uregularization_losses
Уnon_trainable_variables
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Wtrainable_variables
X	variables
жlayers
Жmetrics
вlayer_metrics
 Вlayer_regularization_losses
Yregularization_losses
ьnon_trainable_variables
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
[trainable_variables
\	variables
Ьlayers
№metrics
­layer_metrics
 ыlayer_regularization_losses
]regularization_losses
Ыnon_trainable_variables
р__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
_trainable_variables
`	variables
зlayers
Зmetrics
шlayer_metrics
 Шlayer_regularization_losses
aregularization_losses
эnon_trainable_variables
с__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ctrainable_variables
d	variables
Эlayers
щmetrics
Щlayer_metrics
 чlayer_regularization_losses
eregularization_losses
Чnon_trainable_variables
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
.:,ђ2conv_s_n2d_5/kernel
 :ђ2conv_s_n2d_5/bias
 :	ђ2conv_s_n2d_5/sn
.
g0
h1"
trackable_list_wrapper
5
g0
h1
i2"
trackable_list_wrapper
 "
trackable_list_wrapper
х
jtrainable_variables
k	variables
§layers
■metrics
 layer_metrics
 ђlayer_regularization_losses
lregularization_losses
Ђnon_trainable_variables
у__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ntrainable_variables
o	variables
ѓlayers
Ѓmetrics
ёlayer_metrics
 Ёlayer_regularization_losses
pregularization_losses
єnon_trainable_variables
ж__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
rtrainable_variables
s	variables
Єlayers
ѕmetrics
Ѕlayer_metrics
 іlayer_regularization_losses
tregularization_losses
Іnon_trainable_variables
в__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
.:,ђ@2conv_s_n2d_6/kernel
:@2conv_s_n2d_6/bias
:@2conv_s_n2d_6/sn
.
v0
w1"
trackable_list_wrapper
5
v0
w1
x2"
trackable_list_wrapper
 "
trackable_list_wrapper
х
ytrainable_variables
z	variables
їlayers
Їmetrics
јlayer_metrics
 Јlayer_regularization_losses
{regularization_losses
љnon_trainable_variables
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
}trainable_variables
~	variables
Љlayers
њmetrics
Њlayer_metrics
 ћlayer_regularization_losses
regularization_losses
Ћnon_trainable_variables
№__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ђtrainable_variables
ѓ	variables
ќlayers
Ќmetrics
ўlayer_metrics
 Ўlayer_regularization_losses
Ѓregularization_losses
џnon_trainable_variables
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
-:+@ 2conv_s_n2d_7/kernel
: 2conv_s_n2d_7/bias
: 2conv_s_n2d_7/sn
0
Ё0
є1"
trackable_list_wrapper
8
Ё0
є1
Є2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕtrainable_variables
Ѕ	variables
Џlayers
юmetrics
Юlayer_metrics
 ъlayer_regularization_losses
іregularization_losses
Ъnon_trainable_variables
з__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
їtrainable_variables
Ї	variables
аlayers
Аmetrics
бlayer_metrics
 Бlayer_regularization_losses
јregularization_losses
цnon_trainable_variables
ш__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љtrainable_variables
Љ	variables
Цlayers
дmetrics
Дlayer_metrics
 еlayer_regularization_losses
њregularization_losses
Еnon_trainable_variables
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
-:+ 2conv_s_n2d_8/kernel
:2conv_s_n2d_8/bias
:2conv_s_n2d_8/sn
0
ћ0
Ћ1"
trackable_list_wrapper
8
ћ0
Ћ1
ќ2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќtrainable_variables
ў	variables
фlayers
Фmetrics
гlayer_metrics
 Гlayer_regularization_losses
Ўregularization_losses
«non_trainable_variables
щ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џtrainable_variables
ю	variables
»layers
░metrics
▒layer_metrics
 ▓layer_regularization_losses
Юregularization_losses
│non_trainable_variables
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ъtrainable_variables
а	variables
┤layers
хmetrics
Хlayer_metrics
 иlayer_regularization_losses
Аregularization_losses
Иnon_trainable_variables
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
-:+2conv_s_n2d_9/kernel
:2conv_s_n2d_9/bias
:2conv_s_n2d_9/sn
0
Б0
ц1"
trackable_list_wrapper
8
Б0
ц1
Ц2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
дtrainable_variables
Д	variables
╣layers
║metrics
╗layer_metrics
 ╝layer_regularization_losses
еregularization_losses
йnon_trainable_variables
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
фtrainable_variables
Ф	variables
Йlayers
┐metrics
└layer_metrics
 ┴layer_regularization_losses
гregularization_losses
┬non_trainable_variables
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
«trainable_variables
»	variables
├layers
─metrics
┼layer_metrics
 кlayer_regularization_losses
░regularization_losses
Кnon_trainable_variables
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
ј
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
T
H0
I1
i2
x3
Є4
ќ5
Ц6"
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
.
H0
I1"
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
'
i0"
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
'
x0"
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
(
Є0"
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
(
ќ0"
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
(
Ц0"
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
ѕ2Ё
"__inference__wrapped_model_1471937я
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *NбK
IџF
!і
input_3         
!і
input_4         d
Ы2№
)__inference_model_1_layer_call_fn_1474855
)__inference_model_1_layer_call_fn_1473933
)__inference_model_1_layer_call_fn_1474801
)__inference_model_1_layer_call_fn_1474069└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
D__inference_model_1_layer_call_and_return_conditional_losses_1474444
D__inference_model_1_layer_call_and_return_conditional_losses_1474747
D__inference_model_1_layer_call_and_return_conditional_losses_1473714
D__inference_model_1_layer_call_and_return_conditional_losses_1473796└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_1_layer_call_fn_1474869б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_1_layer_call_and_return_conditional_losses_1474862б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_10_layer_call_fn_1474879б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_1474874б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_2_layer_call_fn_1474893б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_2_layer_call_and_return_conditional_losses_1474886б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_11_layer_call_fn_1474903б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_1474898б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_3_layer_call_fn_1474917б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_3_layer_call_and_return_conditional_losses_1474910б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_5_layer_call_fn_1474931б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_5_layer_call_and_return_conditional_losses_1474924б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_12_layer_call_fn_1474941б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_1474936б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Ц
5__inference_batch_normalization_layer_call_fn_1475010
5__inference_batch_normalization_layer_call_fn_1475023┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474977
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474997┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_4_layer_call_fn_1475037б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_4_layer_call_and_return_conditional_losses_1475030б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_14_layer_call_fn_1475047б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_1475042б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_13_layer_call_fn_1475057б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_1475052б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_reshape_2_layer_call_fn_1475076б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_reshape_2_layer_call_and_return_conditional_losses_1475071б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_reshape_1_layer_call_fn_1475095б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_reshape_1_layer_call_and_return_conditional_losses_1475090б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_concatenate_2_layer_call_fn_1475108б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1475102б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Щ2э
.__inference_conv_s_n2d_5_layer_call_fn_1475324
.__inference_conv_s_n2d_5_layer_call_fn_1475313
.__inference_conv_s_n2d_5_layer_call_fn_1475216
.__inference_conv_s_n2d_5_layer_call_fn_1475205┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2с
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475259
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475151
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475194
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475302┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ќ2ћ
/__inference_up_sampling2d_layer_call_fn_1472292Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
▓2»
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1472286Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┌2О
0__inference_leaky_re_lu_15_layer_call_fn_1475334б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_1475329б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_conv_s_n2d_6_layer_call_fn_1475442
.__inference_conv_s_n2d_6_layer_call_fn_1475431┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475377
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475420┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_up_sampling2d_1_layer_call_fn_1472507Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1472501Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┌2О
0__inference_leaky_re_lu_16_layer_call_fn_1475452б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_1475447б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_conv_s_n2d_7_layer_call_fn_1475549
.__inference_conv_s_n2d_7_layer_call_fn_1475560┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475495
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475538┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_up_sampling2d_2_layer_call_fn_1472722Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1472716Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┌2О
0__inference_leaky_re_lu_17_layer_call_fn_1475570б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_1475565б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_conv_s_n2d_8_layer_call_fn_1475667
.__inference_conv_s_n2d_8_layer_call_fn_1475678┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475613
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475656┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_up_sampling2d_3_layer_call_fn_1472937Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1472931Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┌2О
0__inference_leaky_re_lu_18_layer_call_fn_1475688б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_1475683б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
.__inference_conv_s_n2d_9_layer_call_fn_1475785
.__inference_conv_s_n2d_9_layer_call_fn_1475796┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
л2═
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475774
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475731┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_up_sampling2d_4_layer_call_fn_1473152Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1473146Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┌2О
0__inference_leaky_re_lu_19_layer_call_fn_1475806б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_1475801б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
МBл
%__inference_signature_wrapper_1474125input_3input_4"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 №
"__inference__wrapped_model_1471937╚!%.7<NIFHGgihvxwЁЄєћќЋБЦцXбU
NбK
IџF
!і
input_3         
!і
input_4         d
ф "IфF
D
leaky_re_lu_192і/
leaky_re_lu_19         ђЭИ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474977dHIFG4б1
*б'
!і
inputs         └
p
ф "&б#
і
0         └
џ И
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1474997dIFHG4б1
*б'
!і
inputs         └
p 
ф "&б#
і
0         └
џ љ
5__inference_batch_normalization_layer_call_fn_1475010WHIFG4б1
*б'
!і
inputs         └
p
ф "і         └љ
5__inference_batch_normalization_layer_call_fn_1475023WIFHG4б1
*б'
!і
inputs         └
p 
ф "і         └Ж
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1475102Џjбg
`б]
[џX
*і'
inputs/0          >
*і'
inputs/1          >
ф "-б*
#і 
0          >
џ ┬
/__inference_concatenate_2_layer_call_fn_1475108јjбg
`б]
[џX
*і'
inputs/0          >
*і'
inputs/1          >
ф " і          >┐
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475151rgih;б8
1б.
(і%
inputs          >
p
ф ".б+
$і!
0          >ђ
џ ┐
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475194rgih;б8
1б.
(і%
inputs          >
p 
ф ".б+
$і!
0          >ђ
џ С
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475259ќgihMбJ
Cб@
:і7
inputs+                           
p
ф "@б=
6і3
0,                           ђ
џ С
I__inference_conv_s_n2d_5_layer_call_and_return_conditional_losses_1475302ќgihMбJ
Cб@
:і7
inputs+                           
p 
ф "@б=
6і3
0,                           ђ
џ Ќ
.__inference_conv_s_n2d_5_layer_call_fn_1475205egih;б8
1б.
(і%
inputs          >
p
ф "!і          >ђЌ
.__inference_conv_s_n2d_5_layer_call_fn_1475216egih;б8
1б.
(і%
inputs          >
p 
ф "!і          >ђ╝
.__inference_conv_s_n2d_5_layer_call_fn_1475313ЅgihMбJ
Cб@
:і7
inputs+                           
p
ф "3і0,                           ђ╝
.__inference_conv_s_n2d_5_layer_call_fn_1475324ЅgihMбJ
Cб@
:і7
inputs+                           
p 
ф "3і0,                           ђС
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475377ќvxwNбK
DбA
;і8
inputs,                           ђ
p
ф "?б<
5і2
0+                           @
џ С
I__inference_conv_s_n2d_6_layer_call_and_return_conditional_losses_1475420ќvxwNбK
DбA
;і8
inputs,                           ђ
p 
ф "?б<
5і2
0+                           @
џ ╝
.__inference_conv_s_n2d_6_layer_call_fn_1475431ЅvxwNбK
DбA
;і8
inputs,                           ђ
p
ф "2і/+                           @╝
.__inference_conv_s_n2d_6_layer_call_fn_1475442ЅvxwNбK
DбA
;і8
inputs,                           ђ
p 
ф "2і/+                           @Т
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475495ўЁЄєMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                            
џ Т
I__inference_conv_s_n2d_7_layer_call_and_return_conditional_losses_1475538ўЁЄєMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                            
џ Й
.__inference_conv_s_n2d_7_layer_call_fn_1475549ІЁЄєMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                            Й
.__inference_conv_s_n2d_7_layer_call_fn_1475560ІЁЄєMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                            Т
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475613ўћќЋMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                           
џ Т
I__inference_conv_s_n2d_8_layer_call_and_return_conditional_losses_1475656ўћќЋMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                           
џ Й
.__inference_conv_s_n2d_8_layer_call_fn_1475667ІћќЋMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                           Й
.__inference_conv_s_n2d_8_layer_call_fn_1475678ІћќЋMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                           Т
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475731ўБЦцMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ Т
I__inference_conv_s_n2d_9_layer_call_and_return_conditional_losses_1475774ўБЦцMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ Й
.__inference_conv_s_n2d_9_layer_call_fn_1475785ІБЦцMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Й
.__inference_conv_s_n2d_9_layer_call_fn_1475796ІБЦцMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           Б
D__inference_dense_1_layer_call_and_return_conditional_losses_1474862[%/б,
%б"
 і
inputs         
ф "%б"
і
0          
џ {
)__inference_dense_1_layer_call_fn_1474869N%/б,
%б"
 і
inputs         
ф "і          Б
D__inference_dense_2_layer_call_and_return_conditional_losses_1474886[./б,
%б"
 і
inputs          
ф "%б"
і
0         @
џ {
)__inference_dense_2_layer_call_fn_1474893N./б,
%б"
 і
inputs          
ф "і         @ц
D__inference_dense_3_layer_call_and_return_conditional_losses_1474910\7/б,
%б"
 і
inputs         @
ф "&б#
і
0         ђ
џ |
)__inference_dense_3_layer_call_fn_1474917O7/б,
%б"
 і
inputs         @
ф "і         ђЦ
D__inference_dense_4_layer_call_and_return_conditional_losses_1475030]N0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ }
)__inference_dense_4_layer_call_fn_1475037PN0б-
&б#
!і
inputs         ђ
ф "і         ђц
D__inference_dense_5_layer_call_and_return_conditional_losses_1474924\</б,
%б"
 і
inputs         d
ф "&б#
і
0         └
џ |
)__inference_dense_5_layer_call_fn_1474931O</б,
%б"
 і
inputs         d
ф "і         └Д
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_1474874X/б,
%б"
 і
inputs          
ф "%б"
і
0          
џ 
0__inference_leaky_re_lu_10_layer_call_fn_1474879K/б,
%б"
 і
inputs          
ф "і          Д
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_1474898X/б,
%б"
 і
inputs         @
ф "%б"
і
0         @
џ 
0__inference_leaky_re_lu_11_layer_call_fn_1474903K/б,
%б"
 і
inputs         @
ф "і         @Е
K__inference_leaky_re_lu_12_layer_call_and_return_conditional_losses_1474936Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ Ђ
0__inference_leaky_re_lu_12_layer_call_fn_1474941M0б-
&б#
!і
inputs         ђ
ф "і         ђЕ
K__inference_leaky_re_lu_13_layer_call_and_return_conditional_losses_1475052Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ Ђ
0__inference_leaky_re_lu_13_layer_call_fn_1475057M0б-
&б#
!і
inputs         ђ
ф "і         ђЕ
K__inference_leaky_re_lu_14_layer_call_and_return_conditional_losses_1475042Z0б-
&б#
!і
inputs         └
ф "&б#
і
0         └
џ Ђ
0__inference_leaky_re_lu_14_layer_call_fn_1475047M0б-
&б#
!і
inputs         └
ф "і         └я
K__inference_leaky_re_lu_15_layer_call_and_return_conditional_losses_1475329јJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ Х
0__inference_leaky_re_lu_15_layer_call_fn_1475334ЂJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђ▄
K__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_1475447їIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           @
џ │
0__inference_leaky_re_lu_16_layer_call_fn_1475452IбF
?б<
:і7
inputs+                           @
ф "2і/+                           @▄
K__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_1475565їIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                            
џ │
0__inference_leaky_re_lu_17_layer_call_fn_1475570IбF
?б<
:і7
inputs+                            
ф "2і/+                            ▄
K__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_1475683їIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ │
0__inference_leaky_re_lu_18_layer_call_fn_1475688IбF
?б<
:і7
inputs+                           
ф "2і/+                           ▄
K__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_1475801їIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ │
0__inference_leaky_re_lu_19_layer_call_fn_1475806IбF
?б<
:і7
inputs+                           
ф "2і/+                           Ј
D__inference_model_1_layer_call_and_return_conditional_losses_1473714к!%.7<NHIFGgihvxwЁЄєћќЋБЦц`б]
VбS
IџF
!і
input_3         
!і
input_4         d
p

 
ф "?б<
5і2
0+                           
џ Ј
D__inference_model_1_layer_call_and_return_conditional_losses_1473796к!%.7<NIFHGgihvxwЁЄєћќЋБЦц`б]
VбS
IџF
!і
input_3         
!і
input_4         d
p 

 
ф "?б<
5і2
0+                           
џ Ђ
D__inference_model_1_layer_call_and_return_conditional_losses_1474444И!%.7<NHIFGgihvxwЁЄєћќЋБЦцbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         d
p

 
ф "/б,
%і"
0         ђЭ
џ Ђ
D__inference_model_1_layer_call_and_return_conditional_losses_1474747И!%.7<NIFHGgihvxwЁЄєћќЋБЦцbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         d
p 

 
ф "/б,
%і"
0         ђЭ
џ у
)__inference_model_1_layer_call_fn_1473933╣!%.7<NHIFGgihvxwЁЄєћќЋБЦц`б]
VбS
IџF
!і
input_3         
!і
input_4         d
p

 
ф "2і/+                           у
)__inference_model_1_layer_call_fn_1474069╣!%.7<NIFHGgihvxwЁЄєћќЋБЦц`б]
VбS
IџF
!і
input_3         
!і
input_4         d
p 

 
ф "2і/+                           ж
)__inference_model_1_layer_call_fn_1474801╗!%.7<NHIFGgihvxwЁЄєћќЋБЦцbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         d
p

 
ф "2і/+                           ж
)__inference_model_1_layer_call_fn_1474855╗!%.7<NIFHGgihvxwЁЄєћќЋБЦцbб_
XбU
KџH
"і
inputs/0         
"і
inputs/1         d
p 

 
ф "2і/+                           Ф
F__inference_reshape_1_layer_call_and_return_conditional_losses_1475090a0б-
&б#
!і
inputs         ђ
ф "-б*
#і 
0          >
џ Ѓ
+__inference_reshape_1_layer_call_fn_1475095T0б-
&б#
!і
inputs         ђ
ф " і          >Ф
F__inference_reshape_2_layer_call_and_return_conditional_losses_1475071a0б-
&б#
!і
inputs         └
ф "-б*
#і 
0          >
џ Ѓ
+__inference_reshape_2_layer_call_fn_1475076T0б-
&б#
!і
inputs         └
ф " і          >Ѓ
%__inference_signature_wrapper_1474125┘!%.7<NIFHGgihvxwЁЄєћќЋБЦцiбf
б 
_ф\
,
input_3!і
input_3         
,
input_4!і
input_4         d"IфF
D
leaky_re_lu_192і/
leaky_re_lu_19         ђЭ№
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1472501ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_1_layer_call_fn_1472507ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1472716ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_2_layer_call_fn_1472722ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_1472931ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_3_layer_call_fn_1472937ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1473146ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_4_layer_call_fn_1473152ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ь
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1472286ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_up_sampling2d_layer_call_fn_1472292ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    