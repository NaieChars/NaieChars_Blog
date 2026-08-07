// �������ߣ�����һ�ű�׼����ͼ������������궨

#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // ���̸������ע�� OpenCV �� findChessboardCorners ������"�ڲ��ǵ�"������
    // ���Ǹ������������� 10x7 �����ӣ��ڲ��ǵ��� 9x6 ����ÿ��������1����
    // �����Ȱ� 10x7 ��������ͼ���������궨�������� (9,6) ��Ϊ�ڲ��ǵ�����
    const int squaresX = 10;      // ���������
    const int squaresY = 7;       // ���������
    const int squarePixels = 150; // ÿ��������ͼƬ��ռ�������أ�Խ�������ʾԽ������
    const int borderPixels = 100; // ͼƬ�������ף��������̸����ߵ��½ǵ�������

    int imgWidth = squaresX * squarePixels + 2 * borderPixels;
    int imgHeight = squaresY * squarePixels + 2 * borderPixels;

    // ����ͼ�����ɫ����
    cv::Mat board(imgHeight, imgWidth, CV_8UC1, cv::Scalar(255));

    for (int row = 0; row < squaresY; ++row) 
    {
        for (int col = 0; col < squaresX; ++col) 
        {
            // �������̸����(row+col)Ϊż��ʱͿ��
            if ((row + col) % 2 == 0) 
            {
                int x0 = borderPixels + col * squarePixels;
                int y0 = borderPixels + row * squarePixels;
                cv::Rect square(x0, y0, squarePixels, squarePixels);
                board(square).setTo(cv::Scalar(0));
            }
        }
    }

    std::string outPath = "checkerboard.png";
    cv::imwrite(outPath, board);
    std::cout << "���̸�������: " << outPath << std::endl;
    std::cout << "������(��x��): " << squaresX << "x" << squaresY << std::endl;
    std::cout << "��Ӧ���ڲ��ǵ���(�궨ʱʹ��): " << (squaresX - 1) << "x" << (squaresY - 1) << std::endl;

    return 0;
}