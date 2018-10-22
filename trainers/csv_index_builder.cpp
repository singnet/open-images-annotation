#include <iostream>
#include <unordered_map>
#include <chrono>
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <fstream>

using namespace std;


typedef std::unordered_map<std::string, std::streampos> csv_index_t;


int build_bbox_index_from_csv(csv_index_t &csv_index, std::string fn)
{
    ifstream f(fn);
    if (!f.is_open()) {
        cout << "error opening file " << fn << endl;
        return 1;
    }
    string content;
    std::vector<std::string>   result;
    std::string                line;
    int count = 0;
    int num_cells = 0;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::streampos pos = f.tellg();
    while(std::getline(f, line)) {
        result.clear();
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream, cell, ','))
        {
            result.push_back(cell);
        }
        if (num_cells == 0)
            num_cells = result.size();
        if (num_cells != result.size())
        {
            cout << "error number of cells don't match at line " << count+1 << endl;
            return 1;
        }
        const std::unordered_map<std::string, std::streampos>::iterator it;
        if (csv_index.find(result[0]) == csv_index.end())
        {
            csv_index[result[0]] = pos;
            //cout << result[0] << " " << csv_index[result[0]] << endl;
        }

        count++;
        pos = f.tellg();
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Index built in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " seconds" << std::endl;

    f.close();
    return 0;
}

int get_bbox_from_index(const std::string &img_id, ifstream &f, csv_index_t &csv_index)
{
    csv_index_t::iterator it;
    string line;
    std::vector<string> result;

    it = csv_index.find(img_id);
    if (it == csv_index.end())
        return 1;

    f.seekg(it->second);
    std::getline(f, line);
    result.clear();
    std::stringstream lineStream(line);
    std::string cell;
    while(std::getline(lineStream, cell, ','))
    {
        result.push_back(cell);
    }

    return 0;
}


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "Give the path to Open Images csv of landmarks" << endl;
        cout << "   " << argv[0] << " <PATH TO CSV>" << endl;
        cout << endl;
        return 0;
    }
    const std::string csv_file = argv[1];

    csv_index_t csv_index;
    build_bbox_index_from_csv(csv_index, csv_file);

    ifstream f(csv_file);
    if (get_bbox_from_index("7a8343d5fd73ae0b", f, csv_index) == 0)
        cout << "found" << endl;
    else
        cout << "not found" << endl;

    return 0;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




