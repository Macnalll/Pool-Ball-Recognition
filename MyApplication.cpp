
#include "Utilities.h"


void MyApplication()
{
	char* file_location = "Media/Pool/";
	char* pool_files[] = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"
	};

	Point2f source_points[4], destination_points[4];
	source_points[0] = Point2f( 337.0, 35.0 );
	source_points[1] = Point2f( 1570.0, 21.0 );
	source_points[2] = Point2f( 1888.0, 451.0 );
	source_points[3] = Point2f( 38.0, 472.0 );
	destination_points[0] = Point2f( 0.0, 0.0 );
	destination_points[1] = Point2f( 1434.0, 0.0 );
	destination_points[2] = Point2f( 1434.0, 683.0 );
	destination_points[3] = Point2f( 0.0, 683.0 );

	// Load images
	int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
	Mat* pool_image = new Mat[number_of_images];
	//Mat* new_pool_image = new Mat[number_of_images];
	for (int file_no = 0; (file_no < number_of_images); file_no++)
	{
		cout << pool_files[file_no] << endl;
		string filename(file_location);
		filename.append(pool_files[file_no]);
		pool_image[file_no] = imread(filename, -1);

		if (pool_image[file_no].empty())
		{
			cout << "Could not open " << pool_image[file_no] << endl;
		}
		imshow(pool_files[file_no], pool_image[file_no]);

		Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points), perspective_warped_image;
		perspective_warped_image = Mat::zeros(683, 1434, pool_image[file_no].type());
		warpPerspective(pool_image[file_no], perspective_warped_image, perspective_matrix, perspective_warped_image.size());

		imshow(std::string(pool_files[file_no]) + " perspective", perspective_warped_image);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

