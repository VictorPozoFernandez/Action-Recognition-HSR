$	"ĕ?w?Q@
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
?)?@??U?Z??!ޏ?/?<@*	!       2	!       :$	??BW??P@e???VbW@ZF?=?S??!???N?`@B	!       J	!       R	!       Z	!       b	!       JGPUb qKF?m?W@yJ??E/)@