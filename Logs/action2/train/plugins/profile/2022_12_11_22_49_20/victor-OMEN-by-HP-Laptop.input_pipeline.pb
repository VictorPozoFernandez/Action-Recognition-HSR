$	ȳ˷?L@?
?{??S@?x??[Y??!5&?\R`\@	!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails5&?\R`\@1?G??|?@I??ّ??Z@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?x??[Y??1??z??IB͐*?W??r29*	=
ףp?`@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat/1?闈??!8??ڑ?@@);???R???1l.??9@@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?uʣ??!F???&
B@)?s~?????1??@??b<@:Preprocessing2T
Iterator::Root::ParallelMapV2?"?k$??!?Ds!v2@)?"?k$??1?Ds!v2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceS@?? k??!u?<???@)S@?? k??1u?<???@:Preprocessing2E
Iterator::Root????aN??!ۆ?6rn7@)?ʦ\?}?11	aT?w@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipd?mlv???!I]rc$S@)e?9:Zu?1??? ɮ@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapLl>???!}5?yHuC@)D??<??o?1pu??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??IӠh^?!|? ???)??IӠh^?1|? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?93.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??˒|W@Q??E?6@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	?2???@? ??
I@??z??!?G??|?@*	!       2	!       :$	????H?J@???[?R@B͐*?W??!??ّ??Z@B	!       J	!       R	!       Z	!       b	!       JGPUb q??˒|W@y??E?6@