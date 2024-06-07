import json
import requests
import wordfreq
from collections import OrderedDict
# Sample JSON data

global final_list
final_list = []


def get_all_nodes(input_keyword: str, recursive_level: int = 1,item_limit=40):
    if recursive_level != 0:
        print(f"recursive {recursive_level}, input keyword: {input_keyword}")
        input_json = requests.get('http://api.conceptnet.io/c/en/' + input_keyword+f"?offset=0&limit={item_limit}").json()
        for edge in input_json["edges"]:
            if '/c/en/' in edge["start"]["@id"]:
                keyword = edge["start"]["@id"].replace("/c/en/", "").split("/")[0]
                keyword = keyword.replace("_", " ")
                print(keyword)
                # print(f'{recursive_level}: {keyword}')
                if keyword not in final_list:
                    final_list.append(keyword)
                    get_all_nodes(keyword, recursive_level - 1)
            if '/r/RelatedTo' in edge["@id"] and 'c/en/' in edge["@id"]:
                keyword = edge["@id"].split(",")[-1].split("/")[-2]
                print(f"keyword: {keyword}")
                if keyword not in final_list:
                    final_list.append(keyword)
                    get_all_nodes(keyword, recursive_level - 1)


def get_all_related_words(input_keyword: str, input_relation: str = "RelatedTo", recursive_level: int = 1,item_limit=30):
    """

    :param input_keyword: key word to search
    :param relation: could be RelatedTo, FormOf, IsA, CapableOf,AtLocation ,.etc refer to https://github.com/commonsense/conceptnet5/wiki/Relations
    :param recursive_level:
    :param item_limit: the limit for retrieving related items
    :return:
    """

    if recursive_level != 0:
        print(f"recursive {recursive_level}, input keyword: {input_keyword}")
        api_url = f"https://api.conceptnet.io/c/en/{input_keyword}?rel=/r/{input_relation}&limit={item_limit}"
        input_json = requests.get(api_url).json()
        for edge in input_json["edges"]:
            relation_item=edge["@id"].split(",")[1]
            if 'c/en/' in relation_item:
                print("++++++++ : ",relation_item)
                keyword = relation_item.split("/")[3]
                print(f"keyword: {keyword}")
                if keyword not in final_list and len(keyword) > 1:
                    final_list.append(keyword.replace("_", " "))
                    get_all_related_words(input_keyword=keyword,input_relation=input_relation, recursive_level=recursive_level - 1)


def get_work_frequencies_of_input_word_list(input_word_list: [])->OrderedDict:
    word_set = set(input_word_list)
    my_dict = {}
    for item in word_set:
        my_dict[item]=format(wordfreq.word_frequency(item,"en",'best'),'.10f')

    sorted_dcit=OrderedDict(sorted(my_dict.items(),key=lambda item: item[1],reverse=True))
    print(f"my_dict: {sorted_dcit}")
    for key,value in sorted_dcit.items():
        print(f"{key}: {value}")
    return sorted_dcit

if __name__ == "__main__":

    # Define the theme we want to search and the level we want to dig into recursively
    theme_name="beach"
    dig_level=2
    relation_list=["IsA","AtLocation","RelatedTo","HasProperty","CapableOf","Antonym","SimilarTo","UsedFor"]
    # get_all_nodes(theme_name, dig_level)
    final_list.append(theme_name)
    for relation_item in relation_list:
        get_all_related_words(input_keyword=theme_name,input_relation=relation_item,recursive_level=dig_level)
    output_dict=get_work_frequencies_of_input_word_list(final_list)

    with open(f"{theme_name}.json", "w") as f:
        json.dump(output_dict,f,indent=4)
