
#include "Utilities.h"

// Colour histogram of one ball, using 90 percent of the radius.
// Normalised so closer and further balls are more comparable.
Mat BallHistogram(Mat& hsv_image, Vec3f ball)
{
	Mat ball_mask = Mat::zeros(hsv_image.size(), CV_8U);
	circle(ball_mask, Point(cvRound(ball[0]), cvRound(ball[1])), cvRound(ball[2] * 0.9), Scalar(255), FILLED);

	int channels[] = { 0, 1, 2 };
	int histogram_size[] = { 12, 3, 3 };
	float hue_range[] = { 0, 180 }, saturation_range[] = { 0, 256 }, value_range[] = { 0, 256 };
	const float* ranges[] = { hue_range, saturation_range, value_range };
	Mat histogram;
	calcHist(&hsv_image, 1, channels, ball_mask, histogram, 3, histogram_size, ranges);
	normalize(histogram, histogram, 1, 0, NORM_L1);
	return histogram;
}

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

	// corner points, for perspective transformation
	Point2f source_points[4], destination_points[4];
	source_points[0] = Point2f( 337.0, 35.0 );
	source_points[1] = Point2f( 1570.0, 21.0 );
	source_points[2] = Point2f( 1888.0, 451.0 );
	source_points[3] = Point2f( 38.0, 472.0 );
	destination_points[0] = Point2f( 0.0, 0.0 );
	destination_points[1] = Point2f( 1434.0, 0.0 );
	destination_points[2] = Point2f( 1434.0, 683.0 );
	destination_points[3] = Point2f( 0.0, 683.0 );
	Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points);

	// Integer vector of corner points, for mask
	vector<Point> table_outline =
	{ 
		Point(337, 35),
		Point(1570, 21),
		Point(1888, 451),
		Point(38, 472)
	};

	char* ball_names[16] = { "cue", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };
	int training_row[16] = { 293, 252, 251, 255, 256, 254, 262, 256, 253, 329, 330, 335, 336, 334, 345, 342 };
	int training_column[16] = { 797, 908, 959, 1013, 1072, 1132, 1184, 1244, 1307, 901, 961, 1013, 1067, 1130, 1187, 1245 };
	Mat ball_templates[16];

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
	
		//--------------------------------------------------------------------------------------------------------------------------------------
		// Mask of the table area
		Mat table_mask = Mat::zeros(pool_image[file_no].size(), CV_8U);
		fillConvexPoly(table_mask, table_outline, Scalar(255));
	
		// 2D histogram using hue and saturation, but no value to find green felt under any lighting condition.
		Mat hsv;
		cvtColor(pool_image[file_no], hsv, COLOR_BGR2HSV);
		int channels[] = { 0, 1 };
		int histogram_size[] = { 8, 8 };
		float hue_range[] = { 0, 180 }, saturation_range[] = { 0, 256 };
		const float* ranges[] = { hue_range, saturation_range };
		Mat felt_histogram;
		calcHist(&hsv, 1, channels, table_mask, felt_histogram, 2, histogram_size, ranges);
		normalize(felt_histogram, felt_histogram, 0, 255, NORM_MINMAX);
	
		// Back projection on pool table area. Threshold it to find area of felt.
		Mat back_projection, felt_mask;
		calcBackProject(&hsv, 1, channels, felt_histogram, back_projection, ranges);
		threshold(back_projection, felt_mask, 20, 255, THRESH_BINARY);
		morphologyEx(felt_mask, felt_mask, MORPH_OPEN, Mat::ones(5, 5, CV_8U));

		// Convert felt region to a gray value that allows all other balls to be detected.
		Mat balls_only = pool_image[file_no].clone();
		balls_only.setTo(Scalar(96, 96, 96), felt_mask);
		Mat balls_gray;
		cvtColor(balls_only, balls_gray, COLOR_BGR2GRAY);

		// Hough circles on non perspective image to preserve roundness.
		// Values chosen described in the document.
		vector<Vec3f> found_circles;
		HoughCircles(balls_gray, found_circles, HOUGH_GRADIENT, 1, 22, 130, 14, 18, 27);

		// Save the location of all the found Hough circles.
		vector<Vec3f> balls;
		for (size_t i = 0; i < found_circles.size(); i++)
		{
			Point2f centre(found_circles[i][0], found_circles[i][1]);
			double distance_inside = pointPolygonTest(table_outline, centre, true);
			if (distance_inside >= 8)
			{
				balls.push_back(found_circles[i]);
			}
		}

		// Match each ball to a location from AllBalls.jpg
		if (file_no == 0)
		{
			vector<Point2f> training_table_points, training_image_points;
			for (int ball = 0; ball < 16; ball++)
				training_table_points.push_back(Point2f((float)training_column[ball], (float)training_row[ball]));   // x = column, y = row
			perspectiveTransform(training_table_points, training_image_points, perspective_matrix.inv());
		
			for (int ball = 0; ball < 16; ball++)
			{
				int nearest = 0;
				double nearest_distance = DBL_MAX;
				for (int i = 0; i < (int)balls.size(); i++)
				{
					double distance = norm(Point2f(balls[i][0], balls[i][1]) - training_image_points[ball]);
					if (distance < nearest_distance) { nearest_distance = distance; nearest = i; }
				}
				ball_templates[ball] = BallHistogram(hsv, balls[nearest]);
			}
		}

		// Create a histogram for each ball
		// Back propagate against all learnt balls using Bhattacharryya
		vector<Vec3f> matches;
		for (int i = 0; i < (int)balls.size(); i++)
		{
			Mat histogram = BallHistogram(hsv, balls[i]);
			for (int ball = 0; ball < 16; ball++)
				matches.push_back(Vec3f((float)compareHist(histogram, ball_templates[ball], HISTCMP_BHATTACHARYYA),
					(float)i, (float)ball));
		}

		// Sort each histogram match and join a ball to the most identical histogram
		sort(matches.begin(), matches.end(), [](const Vec3f& a, const Vec3f& b) { return a[0] < b[0]; });
		vector<int> label(balls.size(), -1);
		vector<bool> number_used(16, false);
		for (size_t m = 0; m < matches.size(); m++)
		{
			int i = (int)matches[m][1], ball = (int)matches[m][2];
			if (label[i] == -1 && !number_used[ball])
			{
				label[i] = ball;
				number_used[ball] = true;
			}
		}

		// Draw the Balls on screen and save their position on the table
		Mat balls_image = pool_image[file_no].clone();
		vector<Point2f> image_positions, table_positions;
		for (size_t i = 0; i < balls.size(); i++)
		{
			Point centre(cvRound(balls[i][0]), cvRound(balls[i][1]));
			circle(balls_image, centre, cvRound(balls[i][2]), Scalar(0, 255, 0), 2);
			image_positions.push_back(Point2f(balls[i][0], balls[i][1]));
		}
		perspectiveTransform(image_positions, table_positions, perspective_matrix);

		cout << balls.size() << " balls found" << endl;
		for (size_t i = 0; i < table_positions.size(); i++)
		cout << "  ball " << ((label[i] >= 0) ? ball_names[label[i]] : "?")
			<< " at row=" << cvRound(table_positions[i].y) << " mm, column=" << cvRound(table_positions[i].x) << " mm" << endl;
		
		imshow("back projection", back_projection);
		imshow("felt painted gray", balls_gray);
		imshow("balls", balls_image);

		// Perform perspective transformation of the table
		Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points), perspective_warped_image;
		perspective_warped_image = Mat::zeros(683, 1434, pool_image[file_no].type());
		warpPerspective(pool_image[file_no], perspective_warped_image, perspective_matrix, perspective_warped_image.size());

		imshow(std::string(pool_files[file_no]) + " perspective", perspective_warped_image);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

