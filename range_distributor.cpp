#include <cstdio>
#include <climits>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include <string>
#include <cassert>

// It is just a suggestion to add a Distance function. Use any function you need inside your
// distributor to Key1D and Key2D.

// Forward declaration.
struct Key1D;

// The centroid has a double precision field for greater accuracy.
struct Key1DCentroid {
    double x = 0.0;

    Key1DCentroid& operator=(const Key1DCentroid& other) {
        x = other.x;
        return *this;
    }

    Key1DCentroid& operator=(const Key1D & other);

    const Key1DCentroid operator+(const Key1D& other) const;

    const Key1DCentroid operator-(const Key1D& other) const;

    const Key1DCentroid operator/(const uint64_t divisor) const {
        Key1DCentroid c;
        c.x = x/divisor;
        return c;
    }

    Key1DCentroid operator*(const double multiplier) const {
        Key1DCentroid c;
        c.x = x*multiplier;
        return c;
    }

    bool negative() const {
        return (x < 0);
    }
};

struct Key1D
{
    uint32_t x;

    typedef Key1DCentroid Centroid;

    // Determines affinity to a category.
    double Distance (const Centroid& centroid) {
        auto d = centroid.x - x;
        if (d < 0) {
            d *= -1;
        }
        return d;
    }

    // Populates the dimensions from a buffer read from a file.
    void Load(std::vector<uint32_t>& v, uint32_t idx){
        x = v[idx];
    } 

    // Helps reading the buffer from the file correctly.
    size_t num_dimensions() {
        return 1;
    }

    Key1D& operator=(const Key1D& other) {
        x = other.x;
        return *this;
    }

    bool operator==(const Key1D& other) const {
        return (x == other.x);
    }

    bool operator<(const Key1D& other) const {
        return (x < other.x);
    }

    bool operator>(const Key1D& other) const {
        return (x > other.x);
    }

    const Key1D operator/(uint64_t divisor) const {
        Key1D key;
        key.x = x/divisor;
        return key;
    }

    Key1D operator*(const double multiplier) const {
        Key1D key;
        key.x = x*multiplier;
        return key;
    }
};

Key1DCentroid& Key1DCentroid::operator=(const Key1D & other) {
    x = other.x;
    return *this;
}

const Key1DCentroid Key1DCentroid::operator+(const Key1D& other) const {
    Key1DCentroid c;
    c.x = x + other.x;
    return c;
}

const Key1DCentroid Key1DCentroid::operator-(const Key1D& other) const {
    Key1DCentroid c;
    c.x = x - other.x;
    return c;
}

// Forward declaration.
struct Key2D;

// The centroid has a double precision field for greater accuracy.
struct Key2DCentroid {
    double x;
    double y;

    Key2DCentroid& operator=(const Key2DCentroid& other) {
        x = other.x;
        y = other.y;
        return *this;
    }

    Key2DCentroid& operator=(const Key2D& other);

    const Key2DCentroid operator+(const Key2D& other) const;

    const Key2DCentroid operator-(const Key2D& other) const;

    const Key2DCentroid operator/(const uint64_t divisor) const {
        Key2DCentroid c;
        c.x = x/divisor;
        c.y = y/divisor;
        return c;
    }

    Key2DCentroid operator*(const double multiplier) const {
        Key2DCentroid c;
        c.x = x * multiplier;
        c.y = y * multiplier;
        return c;
    }

    bool negative() const {
        return ((x < 0) || (y < 0));
    }
};

struct Key2D
{
    uint32_t x, y; 
    typedef struct Key2DCentroid Centroid;

    double Distance (const Centroid& centroid) {
        auto dx = centroid.x - x;
        auto dy = centroid.y - y;
        auto d = std::sqrt(dx*dx + dy*dy);
        return d;
    }

    void Load(std::vector<uint32_t>& v, uint64_t idx){
        x = v[idx];
        y = v[idx+1];
    } 

    size_t num_dimensions() {
        return 2;
    }

    bool operator==(const Key2D& other) const {
        return ((x == other.x) && (y == other.y));
    }

    Key2D& operator=(const Key2D& other) {
        x = other.x;
        y = other.y;
        return *this;
    }

    bool operator<(const Key2D& other) const {
        return ((x < other.x) || ((x == other.x) && (y < other.y)));
    }

    bool operator>(const Key2D& other) const {
        return (x > other.x || ((x == other.x) && (y > other.y)));
    }

    const Key2D operator/(uint64_t divisor) const {
        Key2D key;
        key.x = x/divisor;
        key.y = y/divisor;
        return key;
    }

    Key2D operator*(const double multiplier) const {
        Key2D key;
        key.x = x * multiplier;
        key.y = y * multiplier;
        return key;
    }
};

Key2DCentroid& Key2DCentroid::operator=(const Key2D& other) {
    x = other.x;
    y = other.y;
    return *this;
}

const Key2DCentroid Key2DCentroid::operator+(const Key2D& other) const {
    Key2DCentroid c;
    c.x = x + other.x;
    c.y = y + other.y;
    return c;
}

const Key2DCentroid Key2DCentroid::operator-(const Key2D& other) const {
    Key2DCentroid c;
    c.x = x - other.x;
    c.y = y - other.y;
    return c;
}

std::ostream& operator<< (std::ostream& os, const Key1D& key) {
    os << "(" << key.x << ")" << std::endl;
    return os;
}

std::ostream& operator<< (std::ostream& os, const Key1D::Centroid& centroid) {
    os << "(" << centroid.x << ")" << std::endl;
    return os;
}

std::ostream& operator<< (std::ostream& os, const Key2D& key) {
    os << "(" << key.x << ", " << key.y << ")" << std::endl;
    return os;
}

std::ostream& operator<< (std::ostream& os, const Key2D::Centroid& centroid) {
    os << "(" << centroid.x << ", " << centroid.y << ")" << std::endl;
    return os;
}

// This is a category store.
// Each category object stores all the elements contained in this category.
// The elements are stored in a vector since we do not look to maintain any
// order. If needed we will sort the vector.
// Depending on a flush_threshold, once this flush_threshold is reached, 
// cache flushed to a file.
template <typename T>
class Category
{
public:
    Category(uint64_t id, const double learning_rate, 
        size_t flush_threshold = 1000000)
      : category_id_(id), learning_rate_(learning_rate), flush_threshold_(flush_threshold)
      {};

    virtual ~Category(); // do we need this to be virtual?

    // Add a key to this category.
    void Add (const T& t);

    // Potential optimization. TODO
    void AddSubset(const std::vector<T>& s);

    // Remove a key from the category.
    void Yank (const T& t);

    // Potential optimization. TODO
    void YankSubset (const T& l, const T& u, std::vector<T>& out);

    // Pretty print.
    void Print() const ;

    // Given a vector of centroids to other categories, determine if any
    // keys here are better of elsewhere.
    std::vector<std::pair<T, uint32_t>> FindBetterCategory (
        std::shared_ptr<std::vector<typename T::Centroid>> current_centroids) ;

    // Getters.
    const T& GetKeyAt(size_t index) const {
        return category_cache_[index];
    }

    const typename T::Centroid& GetCentroid() const {
        return centroid_;
    }

    const uint64_t& GetCategoryId() const {
        return category_id_;
    }

    const uint64_t& GetNumKeys() const {
        return num_keys_;
    }

    T& GetMax() {
        return max_;
    }

    T& GetMin() {
        return min_;
    }

private:
    void FlushToFile(bool force = false);

    void LoadFromFile();

    void UpdateCache (const T& key, bool removing = false);

private:
    // Category identifier.
    uint64_t category_id_;

    // In memory store for each element.
    std::vector<T> category_cache_;

    // Indicates whether there is data on disk or if this is
    // all resident in memory.
    bool on_disk_ = false;

    // When this threshold is reached the contents are flushed
    // to disk.
    size_t flush_threshold_;

    // Number of keys in this category. Useful to compute the 
    // running average relatively quickly.
    uint64_t num_keys_ = 0;

    // The central point defining the values in this category.
    typename T::Centroid centroid_;

    // Cached External centroids, on which the earning step is executed.
    std::shared_ptr<std::vector<typename T::Centroid>> external_centroids_;

    // Learning rate.
    const double learning_rate_;

    // Maximum value in this category.
    T max_;

    // Minimum value in this category.
    T min_;
};

template<typename T>
Category<T>::~Category() {
    // Remove any files remaining on disk.
   std::ifstream is(std::to_string(category_id_));
    if (is.good()) {
        std::remove ((std::to_string(category_id_)).c_str()); 
    }
}

// Description : Find better categories for all the values in this category, if
//               they exist.
// Input       : Vector of all the centroids of the other categories.
// Output      : Vector of new destination categories for keys.
template<typename T>
std::vector<std::pair<T, uint32_t>> Category<T>::FindBetterCategory (
    std::shared_ptr<std::vector<typename T::Centroid>> external_centroids) {
    assert  ((*external_centroids).size());

    std::vector<std::pair<T, uint32_t>> new_cat_assignments;
    external_centroids_ = external_centroids;

    // XXX OPTIMIZATION
    // Find Better category based on min/max values defined.
    // If (better destination found) { LoadFromFile() and compare each value}
    // If not move to next category.

    // OPTIMIZATION: If the min/max in this category does not indicate the presence
    // of outliers then skip it.
    {
        uint32_t max_distance_to_centroid = GetMax().Distance(centroid_); 
        uint32_t min_distance_to_centroid = GetMin().Distance(centroid_);
        uint32_t upper, lower;
        
        auto iter = begin(*external_centroids);
        auto end_centroids = end(*external_centroids); 

        bool outlier_exists = false;
        for (;iter != end_centroids; ++iter) {
            uint32_t max_distance_to_external_centroid = GetMax().Distance(*iter);
            uint32_t min_distance_to_external_centroid = GetMin().Distance(*iter);
            // If there is any outlier then return
            // max deviation from centroid is "upper"
            // Deviation to external centroid
            //if ((unsigned)(distance_to_external_centroid - lower) <= (upper - lower)) {
            if ((max_distance_to_centroid < max_distance_to_external_centroid) ||
                (min_distance_to_centroid) < min_distance_to_external_centroid) {
                // In range so try again
                continue;
            } else {
                // Outside the range outlier..
                std::cout << "max_distance_to_centroid: " << max_distance_to_centroid << " max_distance_to_external_centroid: " << max_distance_to_external_centroid << " min_distance_to_external_centroid: " << min_distance_to_external_centroid << " min_distance_to_centroid:" << min_distance_to_centroid  << std::endl;
                outlier_exists = true;
            }
        }

        // No use proceeding with scanning this category
        // if no outliers exist.
        if (!outlier_exists) {
            return {};
        }
    }

  
    // Load keys into cache from file.
    LoadFromFile();

    // Iterate through all the keys, compare to the external centroids
    // to determine any new assignments. Could be optimized/batched.
    for (auto& i : category_cache_) {
        auto iter = begin(*external_centroids);
        auto end_centroids = end(*external_centroids); 
        uint32_t new_cat_id = INT_MAX;

        // For each of the external centroids, check if there is a key
        // here that is closer. 
        for (;iter != end_centroids; ++iter) {
            auto external_cat_id = std::distance(begin(*external_centroids), iter);
            if (external_cat_id == GetCategoryId()) {
                // skip the centroid of THIS category.
                continue;
            }

            // My closest centroid is my own centroid.
            uint32_t nearest_centroid_distance = i.Distance(centroid_);
            uint32_t distance_to_external_centroid = i.Distance(*iter);

            // Check for a closer centroid.
            if (distance_to_external_centroid < nearest_centroid_distance) {
                // My closest centroid is the external centroid.
                nearest_centroid_distance = distance_to_external_centroid;
                new_cat_id = external_cat_id;
            }
        }

        // If there is a new category for this key,
        // add it to new_cat_assignments.
        if (new_cat_id != INT_MAX) {
            auto p = std::make_pair(i, new_cat_id); 
            new_cat_assignments.push_back(p);
        }
    }

    return new_cat_assignments;
}

// Flushes category cache contents to a file named
// "category_id".
// This empties the category_cache_.
template<typename T>
void Category<T>::FlushToFile (bool force) {
    if (!force && (category_cache_.size() <= flush_threshold_)) {
        return;
    }

    // Write cache to file.
    std::ofstream os;
    os.open(std::to_string(category_id_), std::ofstream::app);

    for (auto &i : category_cache_) {
        os << i.x << " ";
    }

    os.close();
    
    // Clear cache and flag data presence on file.
    on_disk_ = true;
    category_cache_.clear();
}

// Load on disk category data into the category cache.
// This clears the file of any data, and removes it.
template<typename T>
void Category<T>::LoadFromFile () {
    if (!on_disk_) {
        // Everything is in cache, nothing to do.
        return;
    }

    // Pull up the file contents into a vector.
    std::ifstream is(std::to_string(category_id_));
    if (!is.is_open()) {
        std::cout << "Specified file " << category_id_
                  << " does not exist.\n" ;
        return;
    }

    std::istream_iterator<uint32_t> start(is), end;
    std::vector<uint32_t> from_file(start, end);

    // Merge file with cache contents.
    // Reserve space for all elements coming from the file.
    T t;
    category_cache_.reserve(category_cache_.size() +
        (from_file.size()/t.num_dimensions()));
    
    // Insert each element into the cache.
    for (uint32_t i = 0; i < from_file.size(); i += t.num_dimensions()) {
        T key;
        key.Load(from_file, i);
        category_cache_.push_back(key);
    } 

    is.close();

    // Remove the file to keep cache consistency.
    std::remove(std::to_string(category_id_).c_str());
    on_disk_ = false; // Everything is in cache now.
}

// Add and Yank call this to update the category_cache_.
// If needed the contents are loaded and flushed to a file.
template<typename T>
void Category<T>::UpdateCache (const T& key, bool remove) {

#if _NDEBUG
    if (num_keys_ > 0) {
        double divergence = 0.0;
        uint32_t sum = 0;
        for (auto& i : category_cache_) {
            sum += i.x;
        }
        divergence = (sum/num_keys_) - centroid_.x;
        std::cout << "Divergence is " << divergence << std::endl;
    }
#endif

    // Load everything into the cache.
    // XXX BAD, move this into the REMOVE specific code
    // I dont need to load the entire cache when I ADD anything
    // only when I YANK it.
    //LoadFromFile();

    // Computes a running average, depending on whether the key
    // is being removed or added.
    if (remove) {
        // XXX OPTIMIZATION
        // Have a vector of values marked for removal
        // Update centroid based on what was marked for removal
        // On the next LoadFromFile(), remove all the entries in the list
        // of entries marked for removal.

        LoadFromFile();
        // Compute the running average with the reduced number of keys
        centroid_ = ((centroid_ * num_keys_) - key) / (num_keys_ - 1);
        assert(!centroid_.negative());

        auto iter = std::find(begin(category_cache_), end(category_cache_), key);
        assert (iter != end(category_cache_));
        auto c = category_cache_.erase(iter); 
        num_keys_--;

        // Learning step for added key.
        // This makes the centroid move towards the removed key.
        if (external_centroids_) {
            typename T::Centroid updated_centroid = 
                centroid_ * (1 + learning_rate_) -
                (key * learning_rate_);
            (*external_centroids_)[category_id_] = updated_centroid;
        }
        // learning step
    } else {
        // Compute the runing average with newly added key.
        centroid_ = ((centroid_ * num_keys_) + key) / (num_keys_ + 1);
        category_cache_.push_back(key);
        num_keys_++;
        // Learning step for added key.
        // This makes the centroid move towards the added key.
        if (external_centroids_) {
            typename T::Centroid updated_centroid = 
                centroid_ * (1 - learning_rate_) +
                (key * learning_rate_);
            (*external_centroids_)[category_id_] = updated_centroid;
        }
        // learning step
    }

    // Flush the updated cache into the file.
    FlushToFile();
}
  
// Add an element to the category.
template<typename T>
void Category<T>::Add (const T& t) {

    if (t < min_) {
        min_ = t;
    }

    if (t > max_) {
        max_ = t;
    }

    // Update centroid for this category.
    UpdateCache(t);
};

// Add a range of elements to the category.
template<typename T>
void Category<T>::AddSubset(const std::vector<T>& s) {

}

// Remove key from the category.
template<typename T>
void Category<T>::Yank (const T& t) {
    // Load from file
    LoadFromFile();
    if (t < min_) {
        min_ = t;
    }

    if (t > max_) {
        max_ = t;
    }
    assert (category_cache_.size());
    assert (num_keys_);

    UpdateCache(t, true);
}

// TODO optimization.
template<typename T>
void Category<T>::YankSubset (const T& l, const T& u, std::vector<T>& out){
    if (!on_disk_) {
        return;
    } 
}

template<typename T>
void Category<T>::Print() const {
    std::cout << "Id: " << category_id_;
    std::cout << "\n======================\n";
    for (auto &i : category_cache_) {
        std::cout << i << " " ;
    }
    std::cout << "-----------------------\n";
    std::cout << "Centroid : " << centroid_;
    std::cout << "-----------------------\n\n";
}

/* ========================================================================= */

template<typename T>
class RangeDistributor
{
public:
    RangeDistributor(size_t numCategories, double learningRate);

    uint64_t        GetClassification(T value);
    void            LearnClassifications();
    void            PrintCategories() const {
        for (auto& i : categories_) {
            i.Print();
        }
    }

private:
    const double                       learning_rate_;
    std::vector<Category<T>>           categories_;
    std::shared_ptr<std::vector<typename T::Centroid>>  current_centroids_;
};

template<typename T>
RangeDistributor<T>::RangeDistributor(size_t numCategories,
  double learningRate):
    learning_rate_(learningRate)
{
    current_centroids_ = std::make_shared<std::vector<typename T::Centroid>>();

    // Reserve the vector of categories.
    for (uint32_t id = 0; id < numCategories ; ++id) {
        Category<T> cat(id, learning_rate_);
        categories_.push_back(cat);

        typename T::Centroid centroid;
        (*current_centroids_).push_back(centroid);
    }
}

template<typename T>
inline uint64_t RangeDistributor<T>::GetClassification(T value)
{
    // If a category is not initialized, do so by assigning
    // the centroid to be this value. 
    // XXX BAD, do this in ADD, YANK not here
    for (auto& i : categories_) {
        if (i.GetNumKeys() == 0) {
            i.Add(value);
            (*current_centroids_)[i.GetCategoryId()] = value;
            return i.GetCategoryId();
        }
    }

    // Compute distance to the centroid
    uint32_t min_distance = value.Distance(categories_[0].GetCentroid());
    uint64_t closest_category = 0;
    for (auto& i : categories_) {
        double distance = value.Distance(i.GetCentroid());
        if (distance < min_distance) {
            min_distance = distance;
            closest_category = i.GetCategoryId();
        }
    }

    // Assign the centroid to the category.
    categories_[closest_category].Add(value);
    // Update the current centroids.
    (*current_centroids_)[closest_category] = categories_[closest_category].GetCentroid();
    return closest_category;
}


template<typename T>
void RangeDistributor<T>::LearnClassifications()
{
    for (auto& i : categories_) {
        auto new_cat_assignments = i.FindBetterCategory(current_centroids_);
        uint32_t num_new_assignments = new_cat_assignments.size();

        // Keep re-distributing the keys as long as there are better centroids.
        while (num_new_assignments) {
//             std::cout << "New assignments = " << num_new_assignments << std::endl;
            for (auto& j : new_cat_assignments) {
                i.Yank(j.first); // Key to move (first elem of pair <key, new cat id>).
                categories_[j.second].Add(j.first); // Destination Category id 
                                                    // Add to new category second of pair.
            }

            new_cat_assignments.clear();
            new_cat_assignments = i.FindBetterCategory(current_centroids_);
            num_new_assignments = new_cat_assignments.size();
        }

        // Update the current centroids, this is the basis for the next loop.
        for (auto& c : categories_) {
            (*current_centroids_)[c.GetCategoryId()] = c.GetCentroid();
        }
    }
}

template<typename Generator>
void Graph(size_t categoryCount, size_t numSamples, size_t numGraphs,
           double learningRate, size_t learningWindow, Generator generator)
{
    using ResultType = decltype(generator(0));
    const size_t numSamplesPerGraph = numSamples / numGraphs;
    std::cout << "Starting\n";
    RangeDistributor<ResultType> distributor(categoryCount, learningRate);
    uint32_t cnt = 0;

    std::vector<size_t> categoryCounts(categoryCount, 0);

    while (cnt < numSamples)
    {
        std::cout << "new iteration; \n";
        for (size_t i=0; i<numSamplesPerGraph; ++i, ++cnt)
        {
            const auto key = generator(cnt);
            categoryCounts[distributor.GetClassification(key)]++;

            if (cnt % learningWindow == 0) {
                distributor.LearnClassifications();
            }
        }

        for(auto& count: categoryCounts)
        {
            std::cout << count << " ";
            count = 0;
        }
        std::cout << std::endl;
    }
}

void GraphUniformRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(1337);

    //have a random uniform distribution between 500 - 1290863
    //std::uniform_int_distribution<uint32_t> distribution(500, 1290863);
    std::uniform_int_distribution<uint32_t> distribution(1, 100);
    //Graph(100, 1000000, 10, 0.02, 1000, [&](uint32_t)
    Graph(10, 100000, 10, 0.02, 1000, [&](uint32_t)
    {
        return Key1D{distribution(gen)};
    });
}

void GraphGaussian()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(1337);

    std::normal_distribution<> distribution(50,20);
    //Graph(10, 1000000, 10, 0.1, 1000, [&](uint32_t)
    Graph(10, 100000, 10, 0.1, 1000, [&](uint32_t)
    {
        return Key1D{static_cast<uint32_t>(std::min(100., std::max(0., distribution(gen))))};
    });
}

void GraphTimeseries()
{
    //Graph(1000, 1000000, 10, 0.1, 1000, [](uint32_t i)
    Graph(10, 100000, 10, 0.1, 1000, [](uint32_t i)
    {
        return Key1D{i / 10 + 1466515422};
    });
}

void GraphUniformRandomWithTimeSeries()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(1337);

    //have a random uniform distribution between 500 - 1290863
    std::uniform_int_distribution<uint32_t> distribution(500, 1290863);
    //Graph(100, 1000000, 10, 0.02, 1000, [&](uint32_t i)
    Graph(10, 100000, 10, 0.02, 1000, [&](uint32_t i)
    {
        return Key2D{distribution(gen), i / 1000 + 1466515422};
    });
}

void GraphMonotonicIntegerWithUniformRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(1337);

    //have a random uniform distribution between 500 - 1290863
    std::uniform_int_distribution<uint32_t> distribution(500, 1290863);
    //Graph(100, 1000000, 10, 0.02, 1000, [&](uint32_t i)
    Graph(10, 100000, 10, 0.02, 1000, [&](uint32_t i)
    {
        return Key2D{i / 1000, distribution(gen)};
    });
}

int main(int, char*[])
{

    // 3
    std::cout << "Graph for time series" << std::endl;
    // 1
    std::cout << "Graph for uniform random" << std::endl;
    GraphUniformRandom();

#if 0
    // 2
    std::cout << "Graph for gaussian" << std::endl;
    GraphGaussian();

    // 3
    std::cout << "Graph for time series" << std::endl;
    GraphTimeseries();

    // 4
    std::cout << "Graph for uniform random with time series" << std::endl;
    GraphUniformRandomWithTimeSeries();

    // 5
    std::cout << "Graph for monotonic integer with uniform random" << std::endl;
    GraphMonotonicIntegerWithUniformRandom();
#endif
    return 0;
}

