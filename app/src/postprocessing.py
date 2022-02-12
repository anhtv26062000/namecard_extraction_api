import re
import numpy as np

list_phone = [
    "phone",
    "tel",
    "mobile",
    "đt",
    "sđt",
    "điện thoại",
    "số điện thoại",
    "di động",
]
list_add = [
    "add",
    "address",
    "floor",
    "ward",
    "district",
    "street",
    "city",
    "đc",
    "địa chỉ",
    "khu phố",
    "phường",
    "xã",
    "huyện",
    "đường",
    "quận",
    "tỉnh",
    "thành phố",
    "tphcm",
    "tp.",
    "ha noi",
    "hà nội",
    "hồ chí minh",
]
list_mail = ["@", "mail"]
list_web = ["web", "website", "www"]
# default
# regrex_phone = "((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))"
# edit
regrex_phone = "((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{2,3}[-\.\s]??\d{2,4}[-\.\s]??\d{4}|\(\d{2,3}\)\s*\d{2,4}[-\.\s]??\d{4}|\d{3}))"
# 000-000-0000
# 000 000 0000
# 000.000.0000

# (000)000-0000
# (000)000 0000
# (000)000.0000
# (000) 000-0000
# (000) 000 0000
# (000) 000.0000

# 000-0000
# 000 0000
# 000.0000
# 0000000
# 0000000000
# (000)0000000

# +00 000 000 0000
# +00.000.000.0000
# +00-000-000-0000
# +000000000000
# 0000 0000000000
# 0000-000-000-0000
# 00000000000000
# +00 (000)000 0000
# 0000 (000)000-0000
# 0000(000)000-0000

stopwords_mail = ["email", "mail:", "email:", " ", "e:"]
stopwords_add = [
    "a:",
    "A:",
    "add:",
    "Add:",
    "ADD:",
    "address:",
    "Address:",
    "ADDRESS:",
    "add",
    "Add",
    "ADD",
    "address",
    "Address",
    "ADDRESS",
    "đc:",
    "Đc:",
    "ĐC:",
    "địa chỉ:",
    "Địa chỉ:",
    "Địa Chỉ",
    "ĐỊA CHỈ:",
    "đc",
    "Đc",
    "ĐC",
    "địa chỉ",
    "Địa chỉ",
    "Địa Chỉ",
    "ĐỊA CHỈ",
]
stopwords_web = [
    "w:",
    "W:",
    "web:",
    "Web:",
    "WEB:",
    "website:",
    "Website:",
    "WEBSITE:",
    "web",
    "Web",
    "WEB:",
    "website",
    "Website",
    "WEBSITE",
]


def find_name(rec_result_height_len):
    rec_result_height_len = np.array(rec_result_height_len)
    idx_flat = rec_result_height_len.ravel().argmax()
    idx = np.unravel_index(idx_flat, rec_result_height_len.shape)
    rec_result_height_len = list(rec_result_height_len)
    if rec_result_height_len[idx[0]][1] <= 7:
        rec_result_height_len.pop(idx[0])
        rec_result_height_len = np.array(rec_result_height_len)
        return find_name(rec_result_height_len)
    else:
        return rec_result_height_len[idx[0]]


def name_height_len(rec_result_height_len):
    j = 0
    for i in range(len(rec_result_height_len)):
        if rec_result_height_len[j][1] > 20:
            rec_result_height_len.pop(j)
        else:
            j += 1
    a = list(find_name(rec_result_height_len))

    return a


def post_process(rec_result, rec_result_height_len):
    name = None
    mobile = []
    add = []
    mail = []
    web = []

    name_idx = name_height_len(rec_result_height_len)

    for i, value in enumerate(rec_result):
        querywords = value[1].split()
        if value[2:] == name_idx:
            name = value[1]
        elif (
            any(word in value[1].lower() for word in list_phone)
            or len(str(re.findall(regrex_phone, value[1]))) >= 10
        ):
            res = " ".join(re.findall("[0-9]+", value[1]))
            if res != "":
                mobile.append(" ".join(re.findall("[0-9]+", value[1])))
        elif any(word in value[1].lower() for word in list_add):
            result_adds = [
                word for word in querywords if word.lower() not in stopwords_add
            ]
            res = " ".join(result_adds)
            if res != "":
                add.append(res)
        elif any(word in value[1].lower() for word in list_mail):
            result_mails = [
                word for word in querywords if word.lower() not in stopwords_mail
            ]
            res = "".join(result_mails)
            if res != "":
                mail.append(res)
        elif any(word in value[1].lower() for word in list_web):
            result_webs = [
                word for word in querywords if word.lower() not in stopwords_web
            ]
            res = " ".join(result_webs)
            if res != "":
                web.append(res)

    if len(mobile) > 0:
        mobile = " / ".join(mobile)
    else:
        mobile = None
    if len(add) > 0:
        add = " / ".join(add)
    else:
        add = None
    if len(mail) > 0:
        mail = " / ".join(mail)
    else:
        mail = None
    if len(web) > 0:
        web = " - ".join(web)
    else:
        web = None

    return name, mobile, add, mail, web
