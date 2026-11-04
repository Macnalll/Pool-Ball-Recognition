
#include "Utilities.h"

#define BALL_COUNT 16
#define TABLE_WIDTH 1434
#define TABLE_HEIGHT 683
#define SQUASH 0.65

const vector<vector<Vec3i> > ground_truth = {
	// Table1
	{ { 26, 283, 0 },{ 1090, 314, 1 },{ 1209, 430, 2 },{ 1134, 379, 3 },{ 1051, 335, 4 },{ 1207, 294, 5 },{ 1172, 257, 6 },{ 1174, 351, 7 },{ 1130, 341, 8 },{ 1204, 253, 9 },{ 1170, 317, 10 },{ 1172, 401, 11 },{ 1132, 290, 12 },{ 1210, 383, 13 },{ 1092, 360, 14 },{ 1204, 343, 15 } },
	// Table2
	{ { 1172, 505, 0 },{ 1103, 286, 1 },{ 1226, 624, 2 },{ 1126, 391, 3 },{ 1050, 318, 4 },{ 1306, 310, 5 },{ 622, 209, 6 },{ 1168, 368, 7 },{ 1124, 345, 8 },{ 725, 178, 9 },{ 1271, 184, 10 },{ 796, 663, 11 },{ 438, 298, 12 },{ 1236, 52, 13 },{ 1087, 367, 14 },{ 1204, 399, 15 } },
	// Table3
	{ { 27, 431, 0 },{ 906, 24, 1 },{ 1227, 623, 2 },{ 1203, 444, 3 },{ 32, 337, 4 },{ 1408, 303, 5 },{ 621, 211, 6 },{ 1295, 260, 7 },{ 1252, 228, 8 },{ 725, 180, 9 },{ 1275, 178, 10 },{ 796, 664, 11 },{ 352, 248, 12 },{ 1233, 54, 13 },{ 902, 172, 14 },{ 1204, 402, 15 } },
	// Table4
	{ { 1392, 89, 0 },{ 906, 22, 1 },{ 1407, 543, 2 },{ 1204, 442, 3 },{ 1412, 296, 5 },{ 621, 209, 6 },{ 1289, 253, 7 },{ 1254, 220, 8 },{ 724, 178, 9 },{ 1275, 178, 10 },{ 796, 664, 11 },{ 352, 250, 12 },{ 1235, 54, 13 },{ 902, 170, 14 },{ 1204, 399, 15 } },
	// Table5
	{ { 26, 605, 0 },{ 27, 536, 1 },{ 1408, 546, 2 },{ 1202, 441, 3 },{ 1410, 298, 5 },{ 27, 131, 6 },{ 1289, 251, 7 },{ 1253, 218, 8 },{ 71, 540, 9 },{ 1273, 178, 10 },{ 796, 662, 11 },{ 348, 243, 12 },{ 900, 168, 14 },{ 1202, 400, 15 } },
	// Table6
	{ { 945, 630, 0 },{ 29, 540, 1 },{ 1363, 475, 2 },{ 1202, 441, 3 },{ 1404, 310, 5 },{ 32, 140, 6 },{ 1289, 253, 7 },{ 1252, 218, 8 },{ 70, 538, 9 },{ 1274, 176, 10 },{ 348, 243, 12 },{ 900, 168, 14 },{ 1203, 399, 15 } }
};

// Use Canny edge detection to find largest region, which will be the felt area, rather than the largest green region
// as pool table could be any colour
Mat FeltRegion(Mat& warped, Mat& edges_image)
{
	Mat gray, edges;
	cvtColor(warped, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0);
	Canny(gray, edges, 20, 60);
	dilate(edges, edges, Mat::ones(5, 5, CV_8U));
	edges_image = edges.clone();

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
vector<Vec3f> DetectBalls(Mat& warped, Mat& felt, Mat& balls_gray, Mat& squashed)
{
	// paint the felt gray
	Mat balls_only = warped.clone();
	balls_only.setTo(Scalar(96, 96, 96), felt);
	cvtColor(balls_only, balls_gray, COLOR_BGR2GRAY);

	// minDist 28, Canny 130, accumulator 13, radius 17-26.
	//Mat squashed;
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
	Mat dummy;
	warpPerspective(training_image, training_warped, perspective_matrix, Size(TABLE_WIDTH, TABLE_HEIGHT), INTER_CUBIC);
	Mat training_felt = FeltRegion(training_warped, dummy);
	Mat training_hsv, training_gray;
	cvtColor(training_warped, training_hsv, COLOR_BGR2HSV);
	vector<Vec3f> training_circles = DetectBalls(training_warped, training_felt, training_gray, dummy);

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

	// performance metrics
	int total_tp = 0, total_fp = 0, total_fn = 0, total_correct = 0;

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
		Mat edges;
		warpPerspective(pool_image[file_no], warped, perspective_matrix, Size(TABLE_WIDTH, TABLE_HEIGHT), INTER_CUBIC);
		Mat felt = FeltRegion(warped, edges);
		Mat hsv, balls_gray, squashed;
		cvtColor(warped, hsv, COLOR_BGR2HSV);
		vector<Vec3f> balls = DetectBalls(warped, felt, balls_gray, squashed);

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
		vector<float> label_distance(balls.size(), 2.0f);
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

		if (file_no > 0)
		{
			const vector<Vec3i>& truth = ground_truth[file_no - 1];
			vector<int> order(balls.size());
			for (int i = 0; i < (int)balls.size(); i++) order[i] = i;
			sort(order.begin(), order.end(), [&](int a, int b) { return label_distance[a] < label_distance[b]; });

			vector<bool> truth_used(truth.size(), false);
			int tp = 0, fp = 0, correct = 0;
			for (size_t k = 0; k < order.size(); k++)
			{
				int i = order[k], closest = -1;
				double closest_distance = 25;
				for (int t = 0; t < (int)truth.size(); t++)
				{
					double distance = hypot(balls[i][0] - truth[t][0], balls[i][1] - truth[t][1]);
					if (!truth_used[t] && distance <= closest_distance) { closest_distance = distance; closest = t; }
				}
				if (closest < 0)
					fp++;
				else
				{
					truth_used[closest] = true;
					tp++;
					if (label[i] == truth[closest][2]) correct++;
				}
			}
			int fn = (int)truth.size() - tp;
			cout << "  TP " << tp << ", FP " << fp << ", FN " << fn << ", correctly named " << correct
				<< " of " << truth.size() << endl;
			total_tp += tp; total_fp += fp; total_fn += fn; total_correct += correct;
		}

		// draw on the warped table
		Mat balls_image = warped.clone();
		cout << balls.size() << " balls found" << endl;
		for (size_t i = 0; i < balls.size(); i++)
		{
			Point centre(cvRound(balls[i][0]), cvRound(balls[i][1]));
			int half_width = cvRound(balls[i][2]), half_length = cvRound(balls[i][2] / SQUASH);
			string name = (label[i] >= 0) ? ball_names[label[i]] : "";
			ellipse(balls_image, centre, Size(half_width, half_length), 0, 0, 360, Scalar(0, 255, 0), 2);
			Point text_position = centre + Point(-10, -half_length - 4);
			putText(balls_image, name, text_position, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, Scalar(0, 0, 0), 4);
			putText(balls_image, name, text_position, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
			cout << "  ball " << name << " at row=" << centre.y << " mm, column=" << centre.x << " mm" << endl;
		}

		imshow("squashed", squashed);
		imshow("canny", edges);
		imshow("perspective", warped);
		imshow("felt (largest region)", felt);
		imshow(pool_files[file_no], balls_image);
		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}
}