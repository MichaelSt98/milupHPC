/**
 * @file config_parser.h
 * @brief Config parser.
 *
 * Config parser built on top of boost/filesystem and boost/property_tree.
 * Supported file formats are
 *
 * * `.info`
 * * `.json`
 *
 * @author Johannes Martin
 * @author Michael Staneker
 * @bug no known bugs
 * @todo support more file formats
 */
#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <iostream>
#include <string>
#include <list>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/exception/exception.hpp>
#include <boost/current_function.hpp>
#include <boost/throw_exception.hpp>
#include <boost/foreach.hpp>

//#define BOOST_THROW_EXCEPTION(x) ::boost::throw_exception(x)

/**
 * @brief Config parser class for reading input parameter/settings.
 *
 * A config parser class able to read different input file formats.
 * Currently supported formats are:
 *
 * * `.info`
 * * `.json`
 *
 */
class ConfigParser {
public:

    /**
     * Default constructor.
     */
    ConfigParser();

    /**
     * Constructor.
     *
     * @param file Input config file.
     */
    ConfigParser(const std::string &file);

    /**
     *
     * @param key
     * @return
     */
    std::list<ConfigParser> getObjList(const std::string &key);

    /**
     *
     * @param key
     * @return
     */
    ConfigParser getObj(const std::string &key);

    /**
     *
     * @tparam T
     * @param key
     * @return
     */
    template <typename T>
    T getVal(const std::string &key);

    /**
     *
     * @tparam T
     * @param key
     * @return
     */
    template <typename T>
    std::list<T> getList(const std::string &key);

private:
    // Enable instantiation by ptree to make a recursive usage possible
    /**
     *
     * @param subtree
     */
    ConfigParser(const boost::property_tree::ptree &subtree);

    /// boost property tree instance
    boost::property_tree::ptree tree;
};

/**
 *
 * @tparam T
 * @param key
 * @return
 */
template <typename T>
T ConfigParser::getVal(const std::string &key) {
    return tree.get<T>(key);
}

/**
 *
 * @tparam T
 * @param key
 * @return
 */
template <typename T>
std::list<T> ConfigParser::getList(const std::string &key) {
    std::list <T> lst_;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type &val,tree.get_child(key))
    {
        if (val.second.empty()) { // normal list, just fill it with type T
            lst_.push_back(val.second.get_value<T>());
        } else {
            std::cerr << "List does not contain single values. Please use 'getObjList(const std::string &key)'instead."
                         << " - Returning empty list." << std::endl;
        }
    }
    return lst_;
}

#endif //CONFIGPARSER_H
