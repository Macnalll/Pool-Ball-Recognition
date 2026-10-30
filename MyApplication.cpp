
#include "Utilities.h"

#define BALL_COUNT 16
#define TABLE_WIDTH 1434
#define TABLE_HEIGHT 683
#define SQUASH 0.65 // for making the balls look round

// Use Canny edge detection to find largest region, which will be the felt area, rather than the largest green region
// as pool table could be any colour
Mat FeltRegion(Mat& warped)
{
	Mat gray, edges;
	cvtColor(warped, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0);
	Canny(gray, edges, 20, 60);
	dilate(edges, edges, Mat::ones(5, 5, CV_8U));

	// perform connected component analysis
	Mat not_edges, labels, stats, centroids;
	bitwise_not(edges, not_edges);
	int number_of_regions = connectedComponentsWithStats(not_edges, labels, stats, centroids, 4);
	int largest = 1;
	for (int i = 2; i < number_of_regions; i++)
		if (stats.at<int>(i, CC_STAT_AREA) > stats.at<int>(largest, CC_STAT_AREA))
			largest = i;
	Mat felt = (labels == largest);
	return felt;
}


// find the balls in a perspective image
// use SQUASH to make the balls round
vector<Vec3f> DetectBalls(Mat& warped, Mat& felt, Mat& balls_gray)
{
	// paint the felt gray
	Mat balls_only = warped.clone();
	balls_only.setTo(Scalar(96, 96, 96), felt);
	cvtColor(balls_only, balls_gray, COLOR_BGR2GRAY);

	// minDist 28, Canny 130, accumulator 13, radius 17-26.
	Mat squashed;
	resize(balls_gray, squashed, Size(TABLE_WIDTH, cvRound(TABLE_HEIGHT * SQUASH)), 0, 0, INTER_AREA);
	vector<Vec3f> found_circles, balls;
	HoughCircles(squashed, found_circles, HOUGH_GRADIENT, 1, 28, 130, 13, 17, 26);

	// reject circles in the six pockets (x, y, radius in mm) and too close to the cushions
	const float pockets[6][3] = { { 30, 45, 35 },{ 720, 20, 25 },{ 1405, 45, 35 },
								  { 0, 683, 35 },{ 717, 683, 25 },{ 1434, 683, 35 } };
	const float edge = 12;
	for (size_t i = 0; i < found_circles.size(); i++)
	{
		float x = found_circles[i][0], y = (float)(found_circles[i][1] / SQUASH);
		bool keep = (x >= edge && y >= edge && x <= TABLE_WIDTH - edge && y <= TABLE_HEIGHT - edge);
		for (int p = 0; p < 6 && keep; p++)
			if (hypot(x - pockets[p][0], y - pockets[p][1]) < pockets[p][2])
				keep = false;
		if (keep)
			balls.push_back(Vec3f(x, y, found_circles[i][2]));
	}
	return balls;
}

// colour histogram of one ball, using a hsv histogram,
// normalised so closer and further balls are more comparable
Mat BallHistogram(Mat& hsv_image, Vec3f ball)
{
	Mat ball_mask = Mat::zeros(hsv_image.size(), CV_8U);
	ellipse(ball_mask, Point(cvRound(ball[0]), cvRound(ball[1])),
		Size(max(2, (int)ball[2]), max(2, (int)(ball[2] / SQUASH))), 0, 0, 360, Scalar(255), FILLED);

	int channels[] = { 0, 1, 2 };
	int histogram_size[] = { 16, 3, 3 };
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
	source_points[0] = Point2f(337.0, 35.0);
	source_points[1] = Point2f(1570.0, 21.0);
	source_points[2] = Point2f(1888.0, 451.0);
	source_points[3] = Point2f(38.0, 472.0);
	destination_points[0] = Point2f(0.0, 0.0);
	destination_points[1] = Point2f(1434.0, 0.0);
	destination_points[2] = Point2f(1434.0, 683.0);
	destination_points[3] = Point2f(0.0, 683.0);
	Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points);

	char* ball_names[BALL_COUNT] = { "cue", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };
	int training_row[BALL_COUNT] = { 293, 253, 250, 255, 256, 254, 262, 256, 253, 329, 330, 335, 336, 334, 345, 342 };
	int training_column[BALL_COUNT] = { 797, 908, 959, 1013, 1072, 1132, 1184, 1244, 1307, 901, 961, 1013, 1067, 1130, 1187, 1245 };

	string training_filename(file_location);
	training_filename.append("AllBalls.jpeg");
	Mat training_image = imread(training_filename, -1);
	if (training_image.empty())
	{
		cout << "Could not open " << training_filename << endl;
		return;
	}
	Mat training_warped;
	warpPerspective(training_image, training_warped, perspective_matrix, Size(TABLE_WIDTH, TABLE_HEIGHT), INTER_CUBIC);
	Mat training_felt = FeltRegion(training_warped);
	Mat training_hsv, training_gray;
	cvtColor(training_warped, training_hsv, COLOR_BGR2HSV);
	vector<Vec3f> training_circles = DetectBalls(training_warped, training_felt, training_gray);

	Mat ball_templates[BALL_COUNT];
	for (int ball = 0; ball < BALL_COUNT; ball++)
	{
		Point2f expected((float)training_column[ball], (float)training_row[ball]);
		int nearest = -1;
		double nearest_distance = DBL_MAX;
		for (int i = 0; i < (int)training_circles.size(); i++)
		{
			double distance = norm(Point2f(training_circles[i][0], training_circles[i][1]) - expected);
			if (distance < nearest_distance) { nearest_distance = distance; nearest = i; }
		}
		if (nearest < 0 || nearest_distance > 20)
			cout << "Training: no ball detected near the position of ball " << ball_names[ball] << endl;
		else
			ball_templates[ball] = BallHistogram(training_hsv, training_circles[nearest]);
	}

	// perform transformation, find table, detect and then label balls
	int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
	Mat* pool_image = new Mat[number_of_images];
	for (int file_no = 0; (file_no < number_of_images); file_no++)
	{
		cout << pool_files[file_no] << endl;
		string filename(file_location);
		filename.append(pool_files[file_no]);
		pool_image[file_no] = imread(filename, -1);
		if (pool_image[file_no].empty())
		{
			cout << "Could not open " << filename << endl;
			continue;
		}

		Mat warped;
		warpPerspective(pool_image[file_no], warped, perspective_matrix, Size(TABLE_WIDTH, TABLE_HEIGHT), INTER_CUBIC);
		Mat felt = FeltRegion(warped);
		Mat hsv, balls_gray;
		cvtColor(warped, hsv, COLOR_BGR2HSV);
		vector<Vec3f> balls = DetectBalls(warped, felt, balls_gray);

		// create a histogram for each ball
		// compare each histogram against base values using Bhattacharryya
		vector<Vec3f> matches;
		for (int i = 0; i < (int)balls.size(); i++)
		{
			Mat histogram = BallHistogram(hsv, balls[i]);
			for (int ball = 0; ball < BALL_COUNT; ball++)
				if (!ball_templates[ball].empty())
					matches.push_back(Vec3f((float)compareHist(histogram, ball_templates[ball], HISTCMP_BHATTACHARYYA),
						(float)i, (float)ball));
		}

		// sort each histogram match and join a ball to the most identical histogram
		sort(matches.begin(), matches.end(), [](const Vec3f& a, const Vec3f& b) { return a[0] < b[0]; });
		vector<int> label(balls.size(), -1);
		vector<bool> number_used(BALL_COUNT, false);
		double maximum_distance = 0.8;
		for (size_t m = 0; m < matches.size(); m++)
		{
			if (matches[m][0] > maximum_distance)
				break;
			int i = (int)matches[m][1], ball = (int)matches[m][2];
			if (label[i] == -1 && !number_used[ball])
			{
				label[i] = ball;
				number_used[ball] = true;
			}
		}

		// draw the balls on screen and print their position on the table
		Mat balls_image = warped.clone();
		for (size_t i = 0; i < balls.size(); i++)
		{
			Point centre(cvRound(balls[i][0]), cvRound(balls[i][1]));
			circle(balls_image, centre, cvRound(balls[i][2]), Scalar(0, 255, 0), 2);
		}

		cout << balls.size() << " balls found" << endl;
		for (size_t i = 0; i < balls.size(); i++)
			cout << "  ball " << ((label[i] >= 0) ? ball_names[label[i]] : "?")
				 << " at row=" << cvRound(balls[i][1]) << " mm, column=" << cvRound(balls[i][0]) << " mm" << endl;

		imshow("felt (largest region)", felt);
		imshow(pool_files[file_no], balls_image);
		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}
}