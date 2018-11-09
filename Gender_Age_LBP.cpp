/*
 * Gender_Age_Lbp
 *
 * Contributing Authors: Tome Vang <tome.vang@intel.com>, Neal Smith <neal.p.smith@intel.com>, Heather McCabe <heather.m.mccabe@intel.com>
 *
 *
 *
 */

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "fp16.h"
#include <time.h>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <string.h>

/* Dlib Libraries*/
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dnn.h> // shape_predictor
#include <dlib/image_processing/frontal_face_detector.h> //frontol_face_detector
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/matrix.h>


extern "C"
{
#include <mvnc.h>

}

#define WINDOW_NAME "Ncappzoo Nope"
#define CAM_SOURCE 0
#define XML_FILE "../lbpcascade_frontalface_improved.xml"
// window height and width 16:9 ratio
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 360

// network image resolution
#define NETWORK_IMAGE_WIDTH 160
#define NETWORK_IMAGE_HEIGHT 160

// Location of age and gender networks
//this part directly points to the graph directory
//.//#define GENDER_GRAPH_DIR "../gender_graph/"
#define FACE_GRAPH_DIR "../facenet_celeb_ncs."
//#define AGE_GRAPH_DIR "../age_graph/"
//#define GENDER_CAT_STAT_DIRECTORY "../catstat/Gender/"
//#define AGE_CAT_STAT_DIRECTORY "../catstat/Age/"

// time in seconds to perform an inference on the NCS
#define INFERENCE_INTERVAL 1

using namespace std;
using namespace cv;
using namespace dlib;

// enable networks
//bool enableGenderNetwork = true;
bool enableFaceNetwork = true;
//bool enableAgeNetwork = true;

// text colors and font
const int FONT = cv::FONT_HERSHEY_PLAIN;
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);
const cv::Scalar RED = cv::Scalar(0, 0, 255, 255);
const cv::Scalar PINK = Scalar(255, 80, 180, 255);
const cv::Scalar BLACK = Scalar(0, 0, 0, 255);

// max chars to use for full path.
const unsigned int MAX_PATH = 256;

// opencv cropped face padding. make this larger to increase rectangle size
// default: 60
const int PADDING = 60;
//void *nullptr = NULL;

// the thresholds which the program uses to decide the gender of a person
// default: 0.60 and above is male, 0.40 and below is female
//const float MALE_GENDER_THRESHOLD = 0.60;
//const float FEMALE_GENDER_THRESHOLD = 0.40;


// device setup and preprocessing variables
double networkMean[3];
double networkStd[3];
const uint32_t MAX_NCS_CONNECTED = 1;
uint32_t numNCSConnected = 0;
mvncStatus mvncStat[MAX_NCS_CONNECTED];
const int DEV_NAME_SIZE = 100;
char mvnc_dev_name[DEV_NAME_SIZE];
void* dev_handle[MAX_NCS_CONNECTED];
void* graph_handle[MAX_NCS_CONNECTED];
std::vector<std::string> categories [MAX_NCS_CONNECTED];

typedef unsigned short half_float;

//--------------------------------------------------------------------------------
// // struct for holding age and gender results
//--------------------------------------------------------------------------------
typedef struct networkResults {
//    int gender;
//    float genderConfidence;
//   string ageCategory;
//    float ageConfidence;
//unsigned short *_dst;
//fp16 _dst[128];
float dst[128];

}networkResults;
//mystruct = (struct networkResults){0};


bool preprocess_image(const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat)
{
    // find ratio of to adjust width and height by to make them fit in network image width and height
    double width_ratio = (double)NETWORK_IMAGE_WIDTH / (double)src_image_mat.cols;
    double height_ratio = (double)NETWORK_IMAGE_HEIGHT / (double)src_image_mat.rows;

    // the largest ratio is the one to use for scaling both height and width.
    double largest_ratio = (width_ratio > height_ratio) ? width_ratio : height_ratio;

    // resize the image as close to the network required image dimensions.  After scaling the
     // based on the largest ratio, the resized image will still be in the same aspect ratio as the
    // camera provided but either height or width will be larger than the network required height
    // or width (unless network height == network width.)
    cv::resize(src_image_mat, preprocessed_image_mat, cv::Size(), largest_ratio, largest_ratio, CV_INTER_AREA);

    // now that the preprocessed image is resized, we'll just extract the center portion of it that is exactly the
    // network height and width.
    int mid_row = preprocessed_image_mat.rows / 2.0;
    int mid_col = preprocessed_image_mat.cols / 2.0;
    int x_start = mid_col - (NETWORK_IMAGE_WIDTH/2);
    int y_start = mid_row - (NETWORK_IMAGE_HEIGHT/2);
    cv::Rect roi(x_start, y_start, NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT);
    preprocessed_image_mat = preprocessed_image_mat(roi);
    return true;
}


/*bool read_stat_txt(double* network_mean, double* network_std, const string NETWORK_DIR)
{
    char filename[MAX_PATH];
    strncpy(filename, NETWORK_DIR.c_str(), MAX_PATH);
    strncat(filename, "stat.txt", MAX_PATH);
    FILE* stat_file = fopen(filename, "r");
    if (stat_file == nullptr) {
        return false;
    }
    int num_read_std = 0;
    int num_read_mean = 0;
    num_read_mean = fscanf(stat_file, "%lf%lf%lf\n", &(network_mean[0]), &(network_mean[1]), &(network_mean[2]));
    if (num_read_mean == 3) {
        num_read_std = fscanf(stat_file, "%lf%lf%lf", &(network_std[0]), &(network_std[1]), &(network_std[2]));
    }
    fclose(stat_file);

    if (num_read_mean != 3 || num_read_std != 3) {
        return false;
    }

    for (int i = 0; i < 3; i++) {
        network_mean[i] = 255.0 * network_mean[i];
        network_std[i] = 1.0 / (255.0 * network_std[i]);
    }

    return true;
}

bool read_cat_txt(std::vector<std::string> *categories, const string NETWORK_DIR)
{
    char filename[MAX_PATH];
    strncpy(filename, NETWORK_DIR.c_str(), MAX_PATH);
    strncat(filename, "categories.txt", MAX_PATH);
    FILE* cat_file = fopen(filename, "r");
    if (cat_file == nullptr) {
        return false;
    }

    char cat_line[100];
    fgets(cat_line , 100 , cat_file); // skip the first line
    while (fgets(cat_line , 100 , cat_file) != NULL) {
        if (cat_line[strlen(cat_line) - 1] == '\n')
            cat_line[strlen(cat_line) - 1] = '\0';
        categories->push_back(std::string(cat_line));
    }
    fclose (cat_file);

    if (categories->size() < 1) {
        return false;
    }

    return true;
}

*/
/**
 * @brief read_graph_from_file
 * @param graph_filename [IN} is the full path (or relative) to the graph file to read.
 * @param length [OUT] upon successful return will contain the number of bytes read
 *        which will correspond to the number of bytes in the buffer (graph_buf) allocated
 *        within this function.
 * @param graph_buf [OUT] should be set to the address of a void pointer prior to calling
 *        this function.  upon successful return the void* pointed to will point to a
 *        memory buffer which contains the graph file that was read from disk.  This buffer
 *        must be freed when the caller is done with it via the free() system call.
 * @return true if worked and program should continue or false there was an error.
 */
bool read_graph_from_file(const char *graph_filename, unsigned int *length_read, void **graph_buf)
{
    FILE *graph_file_ptr;

    *graph_buf = nullptr;

    graph_file_ptr = fopen(graph_filename, "rb");
    if (graph_file_ptr == nullptr) {
        return false;
    }

    // get number of bytes in file
    *length_read = 0;
    fseek(graph_file_ptr, 0, SEEK_END);
    *length_read = ftell(graph_file_ptr);
    rewind(graph_file_ptr);

    if(!(*graph_buf = malloc(*length_read))) {
        // couldn't allocate buffer
        fclose(graph_file_ptr);
        return false;
    }

    size_t to_read = *length_read;
    size_t read_count = fread(*graph_buf, 1, to_read, graph_file_ptr);

    if(read_count != *length_read) {
        // didn't read the expected number of bytes
        fclose(graph_file_ptr);
        free(*graph_buf);
        *graph_buf = nullptr;
        return false;
    }
    fclose(graph_file_ptr);

    return true;
}

/**
 * @brief compare result data to sort result indexes
 */
static float *result_data;
/*int sort_results(const void * index_a, const void * index_b) {
    int *a = (int *)index_a;
    int *b = (int *)index_b;
    float diff = result_data[*b] - result_data[*a];
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}
*/


void initNCS(){
    for (int i = 0; i < MAX_NCS_CONNECTED; i ++) {

        // get device name
        mvncStat[i] = mvncGetDeviceName(i, mvnc_dev_name, DEV_NAME_SIZE);
        if (mvncStat[i] != MVNC_OK) {
            if (mvncStat[i] == MVNC_DEVICE_NOT_FOUND) {
                if (i == 0)
                    cout << "Error - Movidius NCS not found, is it plugged in?" << endl;
                numNCSConnected = i;
                break;
            }
            else {
                cout << "Error - mvncGetDeviceName failed: " << mvncStat[i] << endl;
            }
        }
        else {
            cout << "MVNC device " << i << " name: "<< mvnc_dev_name << endl;
        }

        //open device
        mvncStat[i] = mvncOpenDevice(mvnc_dev_name, &dev_handle[i]);
        if (mvncStat[i] != MVNC_OK) {
            cout << "Error - mvncOpenDevice failed: " << mvncStat[i] << endl;
        }
        else {
            cout << "Successfully opened MVNC device" << mvnc_dev_name << endl;
            numNCSConnected++;
        }
    }

    std::cout << "Num of NCS connected: " << numNCSConnected << std::endl;
    //if (numNCSConnected <= 1 && enableAgeNetwork && enableGenderNetwork) {
     //   cout << "Both Age and Gender networks are enabled, but only one NCS device was detected." << endl;
     //   cout << "Please connect two NCS devices or enable only one network (Age or Gender)." << endl;
     //   exit(1);
   // }

}

void initFaceNetwork() {
    // Setup for Gender network
    //if (enableFaceNetwork) {
        // Read the gender stat file
        //if (!read_stat_txt(networkMean, networkStd, GENDER_CAT_STAT_DIRECTORY)) {
         //   cout << "Error - Failed to read stat.txt file for gender network." << endl;
      //      exit(1);
      //  }
        // Read the gender cat file
        //if (!read_cat_txt(&categories[0], GENDER_CAT_STAT_DIRECTORY)) {
        //    cout << "Error - Failed to read categories.txt file for gender network." << endl;
       //     exit(1);
       // }

        // read the gender graph from file:
        char face_graph_filename[MAX_PATH];
        strncpy(face_graph_filename, FACE_GRAPH_DIR, MAX_PATH);
        strncat(face_graph_filename, "graph", MAX_PATH);
        unsigned int graph_len = 0;
        void *face_graph_buf;
        if (!read_graph_from_file(face_graph_filename, &graph_len, &face_graph_buf)) {
            // error reading graph
            cout << "Error - Could not read graph file from disk: " << face_graph_filename << endl;
            mvncCloseDevice(dev_handle[0]);
            exit(1);
        }

        // allocate the graph
        mvncStat[0] = mvncAllocateGraph(dev_handle[0], &graph_handle[0], face_graph_buf, graph_len);
        if (mvncStat[0] != MVNC_OK) {
            cout << "Error - mvncAllocateGraph failed:" << mvncStat[0] << endl;
            exit(1);
        }
        else {
            cout << "Successfully Allocated Face graph for MVNC device." << endl;
        }

    //}
            cout<<"\ninit network returned"<<endl;
}

/*void initAgeNetwork(){

    // Setup for Age network
    if (enableAgeNetwork) {
        if (enableGenderNetwork) {
            // read age stat file
            if (!read_stat_txt(networkMean, networkStd, AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read stat.txt file for age network." << endl;
                exit(1);
            }
            // read cat txt file
            if (!read_cat_txt(&categories[1], AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read categories.txt file for age network." << endl;
                exit(1);
            }

            // read the age graph from file:
            char age_graph_filename[MAX_PATH];
            strncpy(age_graph_filename, AGE_GRAPH_DIR, MAX_PATH);
            strncat(age_graph_filename, "graph", MAX_PATH);
            unsigned int age_graph_len = 0;
            void *age_graph_buf;
            if (!read_graph_from_file(age_graph_filename, &age_graph_len, &age_graph_buf)) {
                // error reading graph
                cout << "Error - Could not read graph file from disk: " << age_graph_filename << endl;
                mvncCloseDevice(dev_handle[1]);
                exit(1);
            }

            // allocate the graph
            mvncStat[1] = mvncAllocateGraph(dev_handle[1], &graph_handle[1], age_graph_buf, age_graph_len);
            if (mvncStat[1] != MVNC_OK) {
                cout << "Error - mvncAllocateGraph failed: %d\n" << mvncStat[1] << endl;
                exit(1);
            }
            else {
                cout << "Successfully Allocated Age graph for MVNC device." << endl;
            }


        } else {
            // if age is the only network selected
            if (!read_stat_txt(networkMean, networkStd, AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read stat.txt file for age network." << endl;
                exit(1);
            }

            if (!read_cat_txt(&categories[0], AGE_CAT_STAT_DIRECTORY)) {
                cout << "Error - Failed to read categories.txt file for network.\n" << endl;
                exit(1);
            }

            // read the age graph from file:
            char age_graph_filename[MAX_PATH];
            strncpy(age_graph_filename, AGE_GRAPH_DIR, MAX_PATH);
            strncat(age_graph_filename, "graph", MAX_PATH);
            unsigned int age_graph_len = 0;
            void *age_graph_buf;
            if (!read_graph_from_file(age_graph_filename, &age_graph_len, &age_graph_buf)) {
                // error reading graph
                cout << "Error - Could not read graph file from disk:" << age_graph_filename << endl;
                mvncCloseDevice(dev_handle[0]);
                exit(1);
            }

            mvncStat[0] = mvncAllocateGraph(dev_handle[0], &graph_handle[0], age_graph_buf, age_graph_len);
            if (mvncStat[0] != MVNC_OK) {
                cout << "Error - mvncAllocateGraph failed: " <<  mvncStat[0] << endl;
                exit(1);
            }
            else {
                cout << "Successfully Allocated Age graph for MVNC device" << endl;
            }
        }
    }
}

*/

float * getInferenceResults(cv::Mat inputMat, std::vector<std::string> networkCategories, mvncStatus ncsStatus, void* graphHandle) {
    cout<<"\nget inferenceresults function reached"<<endl;
    cv::Mat preprocessed_image_mat;
    preprocess_image(inputMat, preprocessed_image_mat);
    if (preprocessed_image_mat.rows != NETWORK_IMAGE_HEIGHT ||
        preprocessed_image_mat.cols != NETWORK_IMAGE_WIDTH) {
        cout << "Error - preprocessed image is unexpected size!" << endl;
        //networkResults error = {0};
        //return error;
        float *error = 0;
        return error;
    }

    // three values for each pixel in the image.  one value for each color channel RGB
    float_t tensor32[3];
    half_float tensor16[NETWORK_IMAGE_WIDTH * NETWORK_IMAGE_HEIGHT * 3];

    uint8_t* image_data_ptr = (uint8_t*)preprocessed_image_mat.data;
    int chan = preprocessed_image_mat.channels();


    int tensor_index = 0;
    for (int row = 0; row < preprocessed_image_mat.rows; row++) {
        for (int col = 0; col < preprocessed_image_mat.cols; col++) {

            int pixel_start_index = row * (preprocessed_image_mat.cols + 0) * chan + col * chan; // TODO: don't hard code

            // assuming the image is in BGR format here
            uint8_t blue = image_data_ptr[pixel_start_index + 0];
            uint8_t green = image_data_ptr[pixel_start_index + 1];
            uint8_t red = image_data_ptr[pixel_start_index + 2];

            //image_data_ptr[pixel_start_index + 2] = 254;

            // then assuming the network needs the data in BGR here.
            // also subtract the mean and multiply by the standard deviation from stat.txt file
            tensor32[0] = (((float_t)blue - networkMean[0]) * networkStd[0]);
            tensor32[1] = (((float_t)green - networkMean[1]) * networkStd[1]);
            tensor32[2] = (((float_t)red - networkMean[2]) * networkStd[2]);

            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[0]))));
            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[1]))));
            tensor16[tensor_index++] =  float2half(*((unsigned*)(&(tensor32[2]))));
        }
    }

    // now convert to array of 16 bit floating point values (half precision) and
    // pass that buffer to load tensor
    ncsStatus = mvncLoadTensor(graphHandle, tensor16, NETWORK_IMAGE_HEIGHT * NETWORK_IMAGE_WIDTH * 3 * sizeof(half_float), nullptr);
      cout<<"\nload tensor just finished"<<endl;
    if (ncsStatus != MVNC_OK) {
        cout << "Error! - LoadTensor failed: " << ncsStatus << endl;
        //networkResults error = {0};
        //return error;
    float *error = 0;
    return error;
      }

    void* result_buf;
    unsigned int res_length;
    void* user_data;
    ncsStatus = mvncGetResult(graphHandle, &result_buf, &res_length, &user_data);
    if (ncsStatus != MVNC_OK) {
        cout << "Error! - GetResult failed: " << ncsStatus << endl;
        //networkResults error = {0};
        //return error;
     float *error = 0;
     return error;
    
     } 

    res_length /= sizeof(unsigned short);
    //float result_fp32[128];
    float* result_fp32 = new float[128];
    fp16tofloat(result_fp32, (unsigned char*)result_buf, res_length);

     //.//not srure if the next part is required or not

    // Sort the results to get the top result
// trying to print out the result_buf
  // for(int j=0; j<res_length; j++){
    //cout<<"\n the buffer value here is %f"<<result_buf[j	]<<endl
    //result_buf++;
  //}
    int indexes[res_length];
    for (unsigned int i = 0; i < res_length; i++) {
        indexes[i] = i;
    }
    result_data = result_fp32;

    networkResults personInferenceResults;

   // if (strcmp(networkCategories[indexes[0]].c_str(), "Male") == 0) {
   //     personInferenceResults.gender = indexes[0];
   //     personInferenceResults.genderConfidence = result_fp32[indexes[0]];
   // }
   // if (strcmp(networkCategories[indexes[0]].c_str(), "0-2") == 0) {
   //     qsort(indexes, res_length, sizeof(*indexes), sort_results);
   //     personInferenceResults.ageCategory = networkCategores[indexes[0]].c_str();
   //     personInferenceResults.ageConfidence = result_fp32[indexes[0]];
   // }


  // for(int i=0; i<res_length; i++){
  //   cout<<"\nvalue %hu"<<networkResults._dst[i]<<endl;
  //  }
for(int i=0; i<res_length-1; i++){
       cout<<"\n value"<<result_fp32[i]<<endl;
       cout<<"\n res_length is "<<res_length<<endl;
       cout<<"\n the present index is "<<i<<endl; 
  }
networkResults nullret = {0};
    return result_fp32;
}


int main (int argc, char** argv) {
    // Opencv variables
    Mat imgIn;
    VideoCapture capture;
    Mat croppedFaceMat;
    Scalar textColor = BLACK;
    Point topLeftRect[5];
    Point bottomRightRect[5];
    Point winTextOrigin;
    CascadeClassifier faceCascade;

    std::vector<Rect> faces;
    //std::vector<matrix<rgb_pixel>> faces;
    String genderText;
    String ageText;
    String rectangle_text;
    clock_t start_time, elapsed_time;
    bool start_inference_timer = true;
    int key;

//variables added from dlib face recognition code
    std::vector<matrix<float,0,1>> Train_face_descriptors;
    //std::vector<matrix<float,0,1>> Test_face_descriptors;
    dlib::matrix<float,128,1> Test_face_descriptors;   
    std::vector<string> Train_Labels;
    float difference; 
    std::vector<int> index;
    std::vector<float>min_diff;

    const int FR_rows = 240;

    deserialize("Train_face_descriptor.dat") >> Train_face_descriptors;
    cout<<"\n deserialize of train face descriptor reached"<<endl;
    deserialize("Train_Labels.dat") >> Train_Labels;
    cout<<"\n deserialize of train labes done";
    
    cout<<"\n just a random par to check the contents of train face descriptors"<<endl;
/*  for (long i=0; i<Train_face_descriptors.size(); i++){
       cout<<"\n Train Descriptor "<<"    "<<i<<endl;
       cout<<Train_face_descriptors[i].nr()<<endl;
       cout<<Train_face_descriptors[i].nc()<<endl;
 }
*/

  for (int i=0; i<=10; i++){
       for(int j=0; j<=128; j++){
    // cout<<Train_face_descriptors[i]<<endl;

     cout<<Train_face_descriptors[i]<<endl;
     //cout<<Train_face_descriptors(i)<<endl;
     cout<<"\n labels now1"<<"    "<<j<<endl;
       }
     cout<<"\n labels now2"<<"     "<<i<<endl;
    } 
  // for(int i=0; i<Train_face_descriptors.nr(); i++){
   // for( int j=0; j<Train_face_descriptors.nc(); j++){
   //     cout<<Train_face_descriptors(i,j)<<endl;
   //     cout<<"label1    "<<j<<endl;
   //  }
   // cout<<"label2   "<<i<<endl
   //}   


    //networkResults currentInferenceResult;
    float *p;
    capture.open(CAM_SOURCE);

    // set the window attributes
    capture.set(CV_CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);

    // create a window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    
    moveWindow(WINDOW_NAME, 0, 0);
    // set a point of origin for the window text
    winTextOrigin.x = 0;
    winTextOrigin.y = 20;

    // Load XML file
    faceCascade.load(XML_FILE);


    // initialize dev handle(s)
    for (int i = 0; i < MAX_NCS_CONNECTED; i++){
        dev_handle[i] = nullptr;
    }

    // if only 1 stick is availble
/*    if (argc == 2) {
        if (strcmp(argv[1], "gender") == 0) {
            enableAgeNetwork = false;
        }
        if (strcmp(argv[1], "age") == 0) {
            enableGenderNetwork = false;
        }
    }
*/
    // initiailze the NCS devices and age and gender networks
    initNCS();
    initFaceNetwork();
    //initGenderNetwork();
   // initAgeNetwork();


    // main loop
    while (true) {
        // feed the capture to the opencv mat
        capture >> imgIn;
        cout<<"\n image capturing started"<<endl;

        // flip the mat horizontally
        flip(imgIn, imgIn, 1);
        cout<<"\n flipping image done"<<endl;
	
        key = waitKey(1);
        // if user presses escape then exit the loop
        if (key == 27)
            break;

        // save rectangle of detected faces to the faces vector
        faceCascade.detectMultiScale(imgIn, faces, 1.1, 2, 0| CASCADE_SCALE_IMAGE, Size(30, 30) );
        cout<<"rectangle of detected faces to the faces vector"<<endl;

        // start timer for inference intervals. Will make an inference every interval. DEFAULT is 1 second.
        if (start_inference_timer) {
        cout<<"start inference part reached"<<endl;
            start_time = clock();
            start_inference_timer = false;
        }

        for(int i=0; i< faces.size(): i++)
	
        // Draw a rectangle and make an inference on each face
        for(int i = 0; i < faces.size(); i++) {
 //the rectangle par tis the the aprt that gets the ball rolling in the first set
            // find the top left and bottom right corners of the rectangle of each face
            cout<<"faces .size part    "<<faces.size()<<"    "<<endl;
            cout<<"drawing of rectangle for face   "<<i<<"     reached"<<endl;
            topLeftRect[i].x = faces[i].x - PADDING;
            topLeftRect[i].y = faces[i].y - PADDING;
            bottomRightRect[i].x = faces[i].x + faces[i].width + PADDING;
            bottomRightRect[i].y = faces[i].y + faces[i].height + PADDING;

            cout<<"printing the values in the faces vector     "<<faces[i].x<<"     "<<faces[i].y<<endl;
            cout<<"int the proces of printin"                                                    
            cout<<"rectangle values are set"<<endl;
            // if the rectangle is within the window bounds, draw the rectangle around the person's face
            if (topLeftRect[i].x > 0 && topLeftRect[i].y > 0 && bottomRightRect[i].x < WINDOW_WIDTH && bottomRightRect[i].y < WINDOW_HEIGHT) {
            cout<<"the rectangle is within the window bounds part reached"<<endl;
            if (topLeftRect[i].x>0 && topLeftRect[i].y > 0 && bottomRightRect[i].x <WINDOW_WIDTH && bottomRightRect[i].y <)
                // draw a rectangle around the detected face
                cv::rectangle(imgIn, topLeftRect[i], bottomRightRect[i], textColor, 2, 8, 0);
               cout<<"rectangle for face    "<<i<<"    done"<<endl;
               cout<<"this is the part that is there after the set of rectangle for face part"

                elapsed_time = clock() - start_time;

                // checks to see if it is time to make inferences
                if ((double)elapsed_time/((double)CLOCKS_PER_SEC) >= INFERENCE_INTERVAL) {
                    cout<<"the rectangel is within the window bounds part reached"<<endl;
                    if(topLeftRect[i].x>0 && topLeftRect)

                if((double)elapse_time/((double)CLOCKS_PER_SEC) >= INFERENCE_INTERVAL){
                    cv::rectangle(imgIn, topLeftRect[i], bo)
                }

                    if(topLeftRect[i].x>0){
                        cout<<"the part where the top left rectangle was reached is now touched upon";
                        cv::rectangle(imgIn, topLeftRect[i],bo)
                    }
                    // crop the face from the webcam feed
                    Rect croppedFaceRect(topLeftRect[i], bottomRightRect[i]);
                    // converts the cropped face rectangle into a opencv mat
                    croppedFaceMat = imgIn(croppedFaceRect);
                    cout<<"\n cropping of face part just finished"<<endl;

                    // process Gender network
    
                    if (enableFaceNetwork) {
                        // send the cropped opencv mat to the ncs device
                        //.///////////currentInferenceResult = getInferenceResults(croppedFaceMat, categories[0], mvncStat[0], graph_handle[0]);
                         p = getInferenceResults(croppedFaceMat, categories[0], mvncStat[0], graph_handle[0]);
                  for ( int i = 0; i < 128; i++ ) {
                      
                      //printf( "*(p + %d) : %d\n", i, *(p + i));
                       cout<<"\n the second print"<<i<<"i is dont"<<endl;
                       cout<<"\n the p value is "<<*(p+i)<<endl; 
                       }
                        // get the appropriate color and text based on the inference results
                        if (currentInferenceResult.genderConfidence >= MALE_GENDER_THRESHOLD) {
                            genderText = categories[0].front().c_str();
                            textColor = BLUE;
                        } else
                        //get the appropriate color and text based on the inference results
                        if (currentInferenceResult.genderConfidence <= FEMALE_GENDER_THRESHOLD){
                            genderText = categories[0].back().c_str();
                           textColor = PINK;
                        } else {
                            genderText = "Unknown";
                            textColor = BLACK;
                        }


                   // here is th for loop for transfer of points
             //std::vector<matrix<float,0,1>> Test_face_descriptors;
             //for(int i = 0; i<128; i++){

//////////////////////////////////////////////
             //.//  Test_face_descriptors[0] = p;
           cout<<"just before passing of values to the test descriptors"<<endl;
           for(int i=0; i<128; i++){
           cout<<"for loop entered"<<endl;
            //Test_face_descriptors(i,0) = *(p+i);
           Test_face_descriptors(0,i) = *(p+i);
            cout<<"value alloted no   "<<i<<endl;
            cout<<"\n here   "<<Test_face_descriptors(0,i);
           }
       
////////////////////////////////////////////////
           cout<<"just before the extra part added is the one her";
           for(int i=0; i<128;i++){
            cout<<"for loop entered<<endl";
            //test_face descriptors(i,0) entered at this point exit ponit is stil unknown and undetermined
            Test_face_descriptors(i,0) = *(p+i);
            cout<<"value aloted no  "<<i<<endl;
            cout<<"\n here "<<Test_face_descriptors(0,i);
           }


             //cout<<"\n the running value is  "<<Test_face_descriptors[i];
              //}
           //



           //direct dlib analogical code starts here

           cout<<"dlib direct analogical code part reached"<<endl;
            std::vector<int> itrIdx;
            bool detect = 0;
            for (size_t i = 0; i < 1; ++i)
            { 
                min_diff.clear(); 
                itrIdx.clear();              
                for(size_t j = 0; j < Train_face_descriptors.size(); ++j)
                {
                    difference = length(Test_face_descriptors-Train_face_descriptors[j]);
                    if (difference <= 0.435)
                    {
                        min_diff.push_back(difference);
                        itrIdx.push_back(j);     
                    }                  
                }
                

                
                      
                if(min_diff.size() != 0)
                {
                	float temp_diff = min_diff[0];
                	int temp_itrIdx= itrIdx[0];
                	detect=0;
                	bool first_diff = 1;
                	for (int m=0; m<min_diff.size(); m++)
                	{
                    	if(min_diff[m] < temp_diff)
                    	{
                        	temp_diff = min_diff[m];
                        	temp_itrIdx = itrIdx[m];
                        	detect = 1;
                        	first_diff = 0;
                    	if(detect == 1) 
                	{                 		     
                    	index.push_back(temp_itrIdx);
                	}
                    else if(first_diff == 1)
                	{                    	
                    	temp_itrIdx = itrIdx[0];                  
                    	index.push_back(temp_itrIdx); 
                	}
                }
                else
                {
                    rectangle_text = "Unknown";
                    putText(imgIn, rectangle_text, topLeftRect[i], FONT, 3, textColor, 3);
                	//cout << "Label: " << rectangle_text << endl;




////////////////part2.....
                cout<<"part 2 is reached"<<endl;

                min_diff.clear(); 
                itrIdx.clear();
                cout<<"min diff and itrldx part reached"<<endl;             
                for(size_t j = 0; j < Train_face_descriptors.size(); ++j)
                {
                    cout<<"for loop of j reached####"<<endl;
                    cout<<"\n\njust before the length function calling"<<endl;
                    difference = length(Test_face_descriptors-Train_face_descriptors[j]);
                    cout<<"\n \n just after the length function call"<<endl;

                     cout<<"here $$$$$ is the difference being printed afer the full execution of the length function. Will print out the differnce in the next line"<<endl;
                   cout<<"difference        "<<difference<<endl;
                    if (difference <= 0.435)
                    {
                      cout<<"difference if condtion reached"<<endl;
                        min_diff.push_back(difference);
                      cout<<"push back of difference done"<<endl;
                        itrIdx.push_back(j);     
                      cout<<"push back of itrldx pointer pushed"<<endl;
                    }                  
                }
                
                cout<<"part 2 is reached"<<endl;

                min_diff.clear();
                itrIdx.clear();
                cout<<"min diff and itrldx part reached"<<endl;
                for(size_t j=0; j<Train_face_descriptors.size(); ++j)
                {
                    cout<<"for loop of j reached "
                }
                cout<<"what in the world was i thinking when i wrote this code";
                cout<<"thsi is whre it all started that is makin the difference";



                
               cout<<"min_diff.size() will be     "<<min_diff.size();       
                if(min_diff.size() != 0)
                for(int i=0;i<result_length_width)
                    cout<<"if min_diff.size()!=0 part reached"<<endl;
                	float temp_diff = min_diff[0];
                	int temp_itrIdx= itrIdx[0];
                	detect=0;
                	bool first_diff = 1;
                	for (int m=0; m<min_diff.size(); m++)
                	{

                    cout<<"first part of the for loop where the running is over min_diff.size()rea"<<endl;
                    	if(min_diff[m] < temp_diff)
                    	{
                            cout<<"comparision of min diff with temp diff reached"<<endl;
                        	temp_diff = min_diff[m];
                        	temp_itrIdx = itrIdx[m];
                        	detect = 1;
                        	first_diff = 0;
                    	}
                	}
                    if(min_diff.size()!=0)
                        for(int i=0;i<result_length_width)
                            cout<<"if min_diff.size()!=0 part has ben reached"<<endl;
                        if(min_diff[m]<min_diff[m]){
                            float temp_diff = min_diff[0];
                            int temp_itrIdx = itrIdx[0];
                            detect =0;
                            bool first_diff= 1;
                            for (int m=0;)
                        }
                        float temp_diff=min_diff[0];
                        int temp_itrIdx= itrIdx[0]
                	if(detect == 1) 
                	{   
                        cout<<"if detect ==1 part reached"<<endl;              		     
                    	index.push_back(temp_itrIdx);
                	}
                    else if(first_diff == 1)
                	{                    	
                        cout<<"if first_diff == 1 part reached"<<endl;
                    	temp_itrIdx = itrIdx[0];                  
                    	index.push_back(temp_itrIdx); 
                	}
                }
                else
                {
                    rectangle_text = "Unknown";
                    cout<<"rectangel text unknown part reached"<<endl;
                    //imshow("hello", imgIn);
                    putText(imgIn, rectangle_text, topLeftRect[i], FONT, 3, textColor, 3);
                	cout << "Label: " << rectangle_text << endl;
                    cout<<"else part finished"<<endl;
                }
direct dlib analogical code ends here


 
                    }
                  

                     process Age network
                    if (enableAgeNetwork) {
                        if (enableGenderNetwork){
                            // send the cropped opencv mat to the ncs device
                            currentInferenceResult = getInferenceResults(croppedFaceMat, categories[1], mvncStat[1], graph_handle[1]);
                        } else {
                            currentInferenceResult = getInferenceResults(croppedFaceMat, categories[0], mvncStat[0], graph_handle[0]);
                            //cout << "Predicted Age: " << ageText << endl;
                            textColor = GREEN;
                        }
                        ageText = currentInferenceResult.ageCategory;
                    }

                    // enable starting the timer again
                    if(enableGenderNetwork){
                        //send the cropped opencv mat to the ncs device
                        currentInferenceResult = getInferenceResults(croppedFaceMat, categories[1], mvncStat[1], mvncStat[1], graph_handle[1]);
                    }
                    else{
                        currentInferenceResult = getInferenceResults (croppedFaceMat, categories[0], mvncStat[0], mvncStat[1], graph_handle[1]);
                        cout<<"Predicted Age:"<<ageText << endl;
                        textColor = GREEN;
                    }
                    ageText = currentInferenceResult.ageCategory;

                    }
                    }
                    cout<<"start_inference_timer reached"<<endl;
                    start_inference_timer = true;
                }
                  //a full paragraph of code that had direct parts coming out of the neural network was neglected and the parts that 
                cout<<"this is the part that reached no to after the double commenting of the extra clause";
                //this is the code that was added as part of the mdn clause that came in after that part as included
                // prepare the gender and age text to be printed to the window
                // rectangle_text = "id: " + to_string(i) + " " + genderText + " " + ageText;
                rectangle_text = genderText + " Hello" + ageText;
                cout<<"rectangle text part is reached"<<endl;
            }
            // print the age and gender text to the window

             //printing part from dlib code
             cout<<"\n index: "<<index.size() << endl;
              cout<<"just after the index part"<<endl;
              cout<<"here is the index i part   "<<index[i];
             // cout<<"the index that has to be fetched from the database is   "<<index[0]<<endl;
            rectangle_text = Train_Labels[index[i]];
              //rectangle_text = Train_Labels[0];
               cout<<"rectangle text part from train_labels[index[i]] part reached"<<endl;
            putText(imgIn, rectangle_text, topLeftRect[i], FONT, 3, textColor, 3);
            cout<<"put text part after the rectangle_text = train_labels part just done"<<endl;
        }

        putText(imgIn,"Press ESC to exit", winTextOrigin, FONT, 2, GREEN, 2);
        // show the opencv mat in the window
        cout<<"imshow part just reached"<<endl;
        imshow(WINDOW_NAME, imgIn);
        cout<<"imshow part is just finished"<<endl;

    } // end main while loop
   cout<<"while loop just ended"<<endl;


    // close all windows
    destroyAllWindows();

    mvncDeallocateGraph(graph_handle[0]);
    mvncCloseDevice(dev_handle[0]);
    if (numNCSConnected > 1) {
        mvncDeallocateGraph(graph_handle[1]);
        mvncCloseDevice(dev_handle[1]);
        //dev handle is the part that came in after the extra additon 
    }
    return 0;
}
