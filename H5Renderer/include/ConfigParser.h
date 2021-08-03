#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <iostream>
#include <string>
#include <list>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/foreach.hpp>

class ConfigParser {
public:
    ConfigParser(const std::string &file="config.json");

    std::list<ConfigParser> getObjList(const std::string &key);
    ConfigParser getObj(const std::string &key);

    template <typename T>
    T getVal(const std::string &key);

    template <typename T>
    std::list<T> getList(const std::string &key);

private:
    // Enable instantiation by ptree to make a recursive usage possible
    ConfigParser(const boost::property_tree::ptree &subtree);

    boost::property_tree::ptree tree;
};

// template function implementations as theses cannot be in source files
template <typename T>
T ConfigParser::getVal(const std::string &key) {
    return tree.get<T>(key);
}

template <typename T>
std::list<T> ConfigParser::getList(const std::string &key) {
    std::list <T> lst_;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &val,tree.get_child(key))
    {
        if(val.second.empty()){
            // normal list, just fill it with type T
            lst_.push_back(val.second.get_value<T>());
        } else {
            std::cerr << "List does not contain sinlge values. Please use 'getObjList(const std::string &key)'instead."
                         << " - Returning empty list." << std::endl;
        }
    }
    return lst_;
}

#endif //CONFIGPARSER_H
