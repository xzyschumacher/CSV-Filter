// c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) list2img.cpp -o list2img$(python3-config --extension-suffix)
// create dynamic link library

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace py = pybind11;
using namespace std;

float get_rms(vector<float> records)
{
    float power_sum = 0;
    for (auto item : records)
    {
        power_sum += item * item;
    }
    return sqrt(power_sum / records.size());
}
// """
// 均方根值 反映的是有效值而不是平均值
// """

float get_hm(vector<float> records)
{
    float h_sum = 0;
    for (auto item : records)
    {
        h_sum += 1.0 / item;
    }
    return records.size() / h_sum;
}

// float get_cv(std::vector<float> records): #标准分和变异系数
//     mean = np.mean(records)
//     std = np.std(records)
//     cv = std / mean
//     return mean, std, cv

vector<vector<float>> deal_list(vector<vector<float>> &mid_sign_list)
{
    vector<vector<float>> mid_sign_img(mid_sign_list.size(), vector<float>(11));
#pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < mid_sign_list.size(); i++)
    {
        if (i % 100000 == 0)
        {
            cout << i << endl;
        }
        vector<float> mid_sign = mid_sign_list[i];
        mid_sign_img[i][0] = mid_sign.size();
        if (mid_sign_img[i][0] == 1)
            continue;
        vector<float>::iterator k = mid_sign.begin();
        mid_sign.erase(k);

        sort(mid_sign.begin(), mid_sign.end());

        int len = mid_sign.size();

        mid_sign_img[i][1] = mid_sign[0];
        mid_sign_img[i][2] = mid_sign[len / 4];
        mid_sign_img[i][3] = mid_sign[len / 2];
        mid_sign_img[i][4] = mid_sign[len * 3 / 4];
        mid_sign_img[i][5] = mid_sign[len - 1];
        mid_sign_img[i][6] = get_rms(mid_sign);
        mid_sign_img[i][7] = get_hm(mid_sign);

        double sum = accumulate(begin(mid_sign), end(mid_sign), 0.0);
        double mean = sum / mid_sign.size(); //均值

        mid_sign_img[i][8] = mean;
        double accum = 0.0;
        for_each(begin(mid_sign), end(mid_sign), [&](const double d)
                 { accum += (d - mean) * (d - mean); });

        mid_sign_img[i][9] = sqrt(accum / mid_sign.size()); //方差
        mid_sign_img[i][10] = mid_sign_img[i][9] / mid_sign_img[i][8];
    }
    return mid_sign_img;
}

// vector<float> deal_list(vector<float> &mid_sign)
// {
//     vector<float> mid_sign_img(11, 0);

//     mid_sign_img[0] = mid_sign.size();

//     if (mid_sign_img[0] == 1)
//         return mid_sign_img;

//     vector<float>::iterator k = mid_sign.begin();
//     mid_sign.erase(k);

//     sort(mid_sign.begin(), mid_sign.end());

//     int len = mid_sign.size();

//     mid_sign_img[1] = mid_sign[0];
//     mid_sign_img[2] = mid_sign[len / 4];
//     mid_sign_img[3] = mid_sign[len / 2];
//     mid_sign_img[4] = mid_sign[len * 3 / 4];
//     mid_sign_img[5] = mid_sign[len - 1];
//     mid_sign_img[6] = get_rms(mid_sign);
//     mid_sign_img[7] = get_hm(mid_sign);

//     double sum = accumulate(begin(mid_sign), end(mid_sign), 0.0);
//     double mean = sum / mid_sign.size(); //均值

//     mid_sign_img[8] = mean;
//     double accum = 0.0;
//     for_each(begin(mid_sign), end(mid_sign), [&](const double d)
//              { accum += (d - mean) * (d - mean); });

//     mid_sign_img[9] = sqrt(accum / (mid_sign.size() - 1)); //方差
//     mid_sign_img[10] = mid_sign_img[9] / mid_sign_img[8];

//     return mid_sign_img;
// }

PYBIND11_MODULE(list2img, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("deal_list", &deal_list, py::return_value_policy::reference);
}