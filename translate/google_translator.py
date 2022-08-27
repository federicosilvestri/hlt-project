import requests

def translate(input: str, lan_to: str, lan_from: str = 'auto') -> str:
    """This function exploits google translate api to translate a sentence from a language to another.

    Args:
        input (str): Input sentence to translate,
        lan_to (str): Label that encode the language symbol of target sentence.
        lan_from (str, optional): Label that encode the language symbol of input sentence. Default value is "auto".

    Returns:
        str: Target sentence.
    """
    input = input.replace("\"", "\\\\\\\"", -1)
    start_pattern = "[[\\\""
    stop_pattern = "\\\","
    res = requests.post(
        "https://translate.google.it/_/TranslateWebserverUi/data/batchexecute?rpcids=MkEWBc&source-path=%2F&f.sid=6697384723060200247&bl=boq_translate-webserver_20220824.02_p0&hl=it&soc-app=1&soc-platform=1&soc-device=1&_reqid=2301709&rt=c",
        headers={
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8"
        },
        data=f'f.req=[[["MkEWBc","[[\\"{input}\\",\\"{lan_from}\\",\\"{lan_to}\\",true],[null]]",null,"generic"]]]'
    ).text
    res = res.replace(input, "", -1)
    res = res[res.index(start_pattern) + len(start_pattern):]
    res = res[:res.index(stop_pattern)]
    res = res.replace("\\\\\\\"", "\"", -1)
    return res

def test():
    """Test"""
    import json
    dataset = json.load(open("../dataset.json", "r"))
    token = "[*]"
    for iter in range((len(dataset) // 40)):
        unique_sentence = ""
        for i in range(40):
            unique_sentence += list(dataset.keys())[iter + i] + token
        res = translate(unique_sentence, lan_to="it", lan_from="en")
        for i, trg in enumerate(res.split(token)[:-1]):
            print(list(dataset.keys())[iter + i], "\t=>\t", trg)