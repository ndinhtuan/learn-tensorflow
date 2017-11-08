import tensorflow as tf 

if __name__ == "__main__":
    
    #get data from Tensor (exist from memory)
    data = tf.random_uniform([4, 5], 0, 100, tf.int32)
    dataset1 = tf.contrib.data.Dataset.from_tensor_slices(data)
    dataset2 = tf.contrib.data.Dataset.from_tensors(data)

    print dataset1 #<TensorSliceDataset shapes: (5,), types: tf.int32> 
                   #Because it is sliced -> [(5,); (5,); (5,); (5,)]
    print dataset2 #<TensorDataset shapes: (4, 5), types: tf.int32>
                   # Because it keep begining state of tensor 'data' => (4, 5)
    print dataset1.output_shapes # (5,)
    print dataset1.output_types # <dtype: 'int32'>
    print dataset2.output_shapes 
    print dataset2.output_types

    #get data from multiple Tensors 
    #Because input of from_tenso_slices(args) is TensorS
    data1 = tf.random_uniform([4]) # need to equal to shape[0] of data 
    dataset3 = tf.contrib.data.Dataset.from_tensor_slices((data, data1))

    print dataset3 #<TensorSliceDataset shapes: ((5,), ()), types: (tf.int32, tf.float32)>
    print dataset3.output_types #(tf.int32, tf.float32)
    print dataset3.output_shapes #TensorShape([Dimension(5)]), TensorShape([])

    # get data from multiple Dataset 
    # using zip(Dataset1, Dataset2, ...) to get corresponding of 
    dataset4 = tf.contrib.data.Dataset.zip([dataset1, dataset2, dataset3])
    
    print dataset4 
    #<ZipDataset shapes: [(5,), (4, 5), ((5,), ())], types: [tf.int32, tf.int32, (tf.int32, tf.float32)]>
    # just one because minnimax element if dataset2 has just 1 element.
    print dataset4.output_shapes
    print dataset4.output_types

    # map current structure to other structure Tensor
    dataset4 = dataset4.map(lambda x, y, z: (y, x, z)) 
    print dataset4 
    print dataset4.output_shapes
    print dataset4.output_types


    #------ Creating an Iterator --------------
    # to access elements from datasets
    # Iterator has 3 kinds of iterator
    # 1. one-shot
    # 2. intializabel
    # 3. reintializabel
    # 4. feedable

    #1. one-shot : push out only 1 data example
    print "Create Iterator"
    dataset5 = tf.contrib.data.Dataset.range(100)
    iterator = dataset5.make_one_shot_iterator() # add object Iterator in graph
    print iterator
    next_element = iterator.get_next() #Tensor Get next object. add this object to graph
    print next_element
    # return next nest struct of Tensor in Dataset
    
    #with tf.Session as sess:
    sess = tf.Session()
    for i in range(4):
        print sess.run(next_element) # run 0 -> 99
    #sess.close()

    #2. initialzabel : allow init number of sample > size of batch <
    # but you need to sess.run explicit iterator.initializer
    max_value = tf.placeholder(dtype=tf.int32, shape=[])
    iterator1 = dataset1.make_initializable_iterator()
    next_element = iterator1.get_next() #object get_next 

    initializer = iterator1.initializer #operation need to run in session 
    # Init iterator over a dataset with 2 (max_value) elements
    # so it is enable you to parameterize the definition of dataset
    sess.run(initializer, feed_dict={max_value: 2})
    for i in range(4):
        print sess.run(next_element)
        #[61 34 64 52 89]
        #[93 89 66 48 88]
        #[88 51 88 37 60]
        #[11  2 57 97 43]
    sess.close()

    #3. reintializable
    #

    


