import xml.etree.ElementTree as ET

if __name__ == "__main__":
    my_file = open("datasets/RDD2020_filtered/ImageSets/train.txt", "r")
    content_list = my_file.readlines()
    my_file.close()
    d00 = []
    d10 = []
    d20 = []
    d40 = []
    print('Number images old:', len(content_list))

    for img in content_list:
        img_ = img.rstrip("\n")
        anno = ET.parse('datasets/RDD2020_filtered/Annotations/' + img_ + '.xml')
        anno = anno.getroot()
        # get image ids for each label
        for child in anno:
            for childchild in child:
                if childchild.text == 'D00':
                    d00.append(img)
                if childchild.text == 'D10':
                    d10.append(img)
                if childchild.text == 'D20':
                    d20.append(img)
                if childchild.text == 'D40':
                    d40.append(img)
    print('D00:', len(d00))
    print('D10:', len(d10))
    print('D20:', len(d20))
    print('D40:', len(d40))
    # drop duplicates in the list
    d10 = list(set(d10))
    # add images that have a D10 label to the dataset again to double its number
    content_list.extend(d10)

    # just to print the made changes
    print('\nNumber images new:', len(content_list))
    d00 = []
    d10 = []
    d20 = []
    d40 = []
    for img in content_list:
        img_ = img.rstrip("\n")
        anno = ET.parse('datasets/RDD2020_filtered/Annotations/' + img_ + '.xml')
        anno = anno.getroot()
        for child in anno:
            for childchild in child:
                if childchild.text == 'D00':
                    d00.append(img)
                if childchild.text == 'D10':
                    d10.append(img)
                if childchild.text == 'D20':
                    d20.append(img)
                if childchild.text == 'D40':
                    d40.append(img)
    print('D00:', len(d00))
    print('D10:', len(d10))
    print('D20:', len(d20))
    print('D40:', len(d40))

    with open("datasets/RDD2020_filtered/ImageSets/train_oversampling.txt", 'w') as f:
        for sample in content_list:
            f.write(sample)