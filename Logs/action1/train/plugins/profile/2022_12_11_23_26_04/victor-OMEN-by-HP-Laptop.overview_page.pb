?$	"ĕ?w?Q@
{py?X@
???C??!-σ?3a@	!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-σ?3a@1ޏ?/?<@I???N?`@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!
???C??1??U?Z??IZF?=?S??r29*	.?$?J?@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapx?'-\V??!?"V?QL@)@M-[????1?????G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??-Y???!,\?m??=@)?W)?k??1 ?G??]7@:Preprocessing2T
Iterator::Root::ParallelMapV2????3???!f#?J?	@)????3???1f#?J?	@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??̒ 5??!??R?%?@)?8???֤?1!8d??@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????ե?!?l??UE@)?wg????1?Cc?o?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipM.??:???!+???N@)?聏????1?V?8J?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatl??TO???!?ί??0@)???k?6??1 ?4?? @:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???t=х?!?Б?!@??)???t=х?1?Б?!@??:Preprocessing2E
Iterator::Root?f?|?|??!?3y3#!@)?r??+|?1"-1'???:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate[0]::TensorSlicevS?k%tw?!@???(8??)vS?k%tw?1@???(8??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????p?!sF??1g??)?????p?1sF??1g??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}]?! '?{????)??H?}]?1 '?{????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[3]::Concatenate?y??Q}?!?Q??i???)3j?J>vW?1?ٳ5?:??:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[1]::FromTensor{/?h?G?!????X??){/?h?G?1????X??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?95.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIKF?m?W@QJ??E/)@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	??\??Z@`
?)?@??U?Z??!ޏ?/?<@*	!       2	!       :$	??BW??P@e???VbW@ZF?=?S??!???N?`@B	!       J	!       R	!       Z	!       b	!       JGPUb qKF?m?W@yJ??E/)@?"^
Csequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/splitSplit???????!???????"?
?gradient_tape/sequential/lstm/while/sequential/lstm/while_grad/body/_1820/gradient_tape/sequential/lstm/while/gradients/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2?o?V=҅?!?;?캘?"b
Dsequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/MatMulMatMulH:??/??!?lD_???0"f
Ksequential/lstm_1/while/body/_354/sequential/lstm_1/while/lstm_cell_1/splitSplit?$&?=ځ?!?u??_??"f
Ksequential/lstm_2/while/body/_717/sequential/lstm_2/while/lstm_cell_2/splitSplit??Zҁ?!?W6?ԩ?"?
?gradient_tape/sequential/lstm_2/while/sequential/lstm_2/while_grad/body/_1080/gradient_tape/sequential/lstm_2/while/gradients/sequential/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2?\??铁?!
??9??"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_1450/gradient_tape/sequential/lstm_1/while/gradients/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2???r ??! ۸6?@??"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_1450/gradient_tape/sequential/lstm_1/while/gradients/AddN_6AddN$7??=?~?!r???+??"b
Fsequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/MatMul_2MatMul???9?r}?!K|wE???"b
Fsequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/MatMul_1MatMulJ?~f?|?!`?b??ж?Q      Y@YHT?n???a?՘H?X@"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?95.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 