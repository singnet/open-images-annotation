#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

// Same network structure as dnn_mmod_detection_ex.cpp
// we can't change this without needing to create a new python binding :-(

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<bn_con<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>;

struct training_sample
{
    matrix<rgb_pixel> input_image;
    std::vector<mmod_rect> boxes;
};

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "Give the path to where the dlib xml files are " << endl;
        cout << "   " << argv[0] << " <dlib xml path>" << endl;
        cout << endl;
        return 0;
    }
    const std::string oi_directory = argv[1];

    using namespace dlib::image_dataset_metadata;
    // TODO add validation data
    dataset training_data;
    dataset testing_data;

    //load_image_dataset(images_train, face_boxes_train, oi_directory+"/training.xml");
    load_image_dataset_metadata(training_data, oi_directory+"/training.xml");
    load_image_dataset_metadata(testing_data, oi_directory+"/testing.xml");

    cout << "num training images: " << training_data.images.size() << endl;
    cout << "num testing images:  " << testing_data.images.size() << endl;

    // Despite loading training data on the fly, we need to load all the boxes because that's
    // how mmod_options is initialized.
    std::vector<std::vector<mmod_rect>> preload_boxes;

    std::vector<mmod_rect> rects;
    for (unsigned long i = 0; i < std::min((size_t) 10000, training_data.images.size()); ++i)
    {
        double min_rect_size = std::numeric_limits<double>::infinity();
        rects.clear();
        for (unsigned long j = 0; j < training_data.images[i].boxes.size(); ++j)
        {
            if (!training_data.images[i].boxes[j].ignore)
            {
                rects.push_back(mmod_rect(training_data.images[i].boxes[j].rect));
                min_rect_size = std::min<double>(min_rect_size, rects.back().rect.area());
            }
            rects.back().label = training_data.images[i].boxes[j].label;
        }

        if (rects.size() != 0)
        {
            preload_boxes.push_back(rects);
        }
    }

    cout << "Preloaded boxes" << endl;

    mmod_options options(preload_boxes, 70,50);
    preload_boxes.clear();

    cout << "num detector windows: "<< options.detector_windows.size() << endl;
    for (auto& w : options.detector_windows)
        cout << "detector window width by height: " << w.width << " x " << w.height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;

    net_type net(options);
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("oi_face_detect_mmod_sync", std::chrono::seconds(600));
    trainer.set_iterations_without_progress_threshold(30000);
    trainer.set_test_iterations_without_progress_threshold(1000);

    dlib::rand rnd;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    cout << "Creating pipe" << endl;
    dlib::pipe<training_sample> training_data_pipe(200);
    dlib::pipe<training_sample> testing_data_pipe(200);

    auto f = [](time_t seed, dataset &data, dlib::pipe<training_sample> &data_pipe)
    {
        dlib::rand rnd(time(0)+seed);

        matrix<rgb_pixel> input_image;
        std::vector<mmod_rect> input_boxes;

        random_cropper cropper;
        // we will use defaults used instead of these options:
        //cropper.set_chip_dims(200, 200);
        //cropper.set_min_object_size(40,40);
        training_sample temp;

        while(data_pipe.is_enabled())
        {
            temp.boxes.clear();
            input_boxes.clear();

            // Pick a random input image.
            const image& image_info = data.images[rnd.get_random_32bit_number()%data.images.size()];

            // Load the input image.
            load_image(input_image, image_info.filename);
            for (uint i = 0; i < image_info.boxes.size(); i++)
                input_boxes.push_back(mmod_rect(image_info.boxes[i].rect));

            cropper(input_image, input_boxes, temp.input_image, temp.boxes);
            disturb_colors(temp.input_image, rnd);

            // Push the result to be used by the trainer.
            data_pipe.enqueue(temp);
        }
    };
    std::thread data_loader1([f, &training_data, &training_data_pipe](){ f(1, training_data, training_data_pipe); });
    std::thread data_loader2([f, &training_data, &training_data_pipe](){ f(2, training_data, training_data_pipe); });
    std::thread data_loader3([f, &training_data, &training_data_pipe](){ f(3, training_data, training_data_pipe); });
    std::thread data_loader4([f, &training_data, &training_data_pipe](){ f(4, training_data, training_data_pipe); });
    std::thread data_loader5([f, &testing_data, &testing_data_pipe](){ f(5, testing_data, testing_data_pipe); });
    cout << "Threads started" << endl;

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<std::vector<mmod_rect>> labels; 

    std::vector<matrix<rgb_pixel>> samples_test;
    std::vector<std::vector<mmod_rect>> labels_test; 

    auto get_test_samples = [&testing_data_pipe, &samples_test, &labels_test]()
    {
        training_sample temp;
        samples_test.clear();
        labels_test.clear();
        while (samples_test.size() < 50)
        {
            testing_data_pipe.dequeue(temp);
            samples_test.push_back(std::move(temp.input_image));
            labels_test.push_back(std::move(temp.boxes));
            if (0)//samples_test.size() < 20)
            {
                image_window win;
                win.set_image(samples_test.back());
                for (uint i = 0; i < labels_test.back().size(); i++) {
                    win.add_overlay(labels_test.back()[i]);
                }
                cin.get();
            }
        }
    };

    uint64_t batches = 0;
    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-5.
    while(trainer.get_learning_rate() >= 1e-5)
    {
        samples.clear();
        labels.clear();

        training_sample temp;
        // make a 50-image mini-batch
        while(samples.size() < 50)
        {
            training_data_pipe.dequeue(temp);
            samples.push_back(std::move(temp.input_image));
            labels.push_back(std::move(temp.boxes));
        }
        trainer.train_one_step(samples, labels);
        batches++;

        if (batches > 0 && batches % 20 == 0)
        {
            get_test_samples();
            trainer.test_one_step(samples_test, labels_test);
        }

    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    training_data_pipe.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("mmod_network.dat") << net;

    training_sample temp;
    samples_test.clear();
    labels_test.clear();
    while (samples_test.size() < 50)
    {
        testing_data_pipe.dequeue(temp);
        samples_test.push_back(std::move(temp.input_image));
        labels_test.push_back(std::move(temp.boxes));
        if (0) //samples_test.size() < 20)
        {
            image_window win;
            win.set_image(samples_test.back());
            for (uint i = 0; i < labels_test.back().size(); i++) {
                win.add_overlay(labels_test.back()[i]);
            }
            cin.get();
        }
    }
    cout << "training results: " << test_object_detection_function(net, samples_test, labels_test) << endl;

    testing_data_pipe.disable();
    data_loader5.join();

    // If you are running many experiments, it's also useful to log the settings used
    // during the training experiment.  This statement will print the settings we used to
    // the screen.
    cout << trainer << endl;

    // Now lets run the detector on the testing images and look at the outputs.  
    image_window win;
    for (auto&& img : samples_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    return 0;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




