$	FaEP@?^"?%V@Ȗ??2???!0,?-?_@	!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,?-?_@1?ɐck@I(?r?w^@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!Ȗ??2???1)?Ǻ???Io?m??r29*?C?l??k@)       =2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatekծ	i??!?g[???F@)?o??R???1?G????@:Preprocessing2T
Iterator::Root::ParallelMapV2Z?!?[??!?-+U??2@)Z?!?[??1?-+U??2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???????!De?m?*@)???????1De?m?*@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatG????g??!?z????+@)???ל?1%T".?)@:Preprocessing2E
Iterator::Root?\???!??ȨC?=@)
?????1"?:?2C%@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?&??@??!???)X?J@)???.\??1oc??+!@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipj??????!`??o?Q@)uۈ'?y?1????i?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??f?v?d?!?{??{4??)??f?v?d?1?{??{4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?94.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?bf8?W@Q?ՙy~@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	2Wզ@-??l9@)?Ǻ???!?ɐck@*	!       2	!       :$	jO?9?oN@O?U)?T@o?m??!(?r?w^@B	!       J	!       R	!       Z	!       b	!       JGPUb q?bf8?W@y?ՙy~@