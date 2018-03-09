task2index = {'skirt_length_labels': 0,
              'coat_length_labels': 1,
              'collar_design_labels': 2,
              'lapel_design_labels': 3,
              'neck_design_labels': 4,
              'neckline_design_labels': 5,
              'pant_length_labels': 6,
              'sleeve_length_lables': 7}

index2task = {v : k for k, v in task2index.items()}

label2index = {}

skirt_length_labels = {'Invisible': 0,
                       'Short Length': 1,
                       'Knee Length': 2,
                       'Midi Length': 3,
                       'Ankle Length': 4,
                       'Floor Length': 5}
label2index[0] = skirt_length_labels

coat_length_labels = {'Invisible': 0,
                      'High Waist Length': 1,
                      'Regular Length': 2,
                      'Long Length': 3,
                      'Micro Length': 4,
                      'Knee Length': 5,
                      'Midi Length': 6,
                      'Ankle&Floor Length': 7}
label2index[1] = coat_length_labels

collar_disign_labels = {'Invisible': 0,
                        'Shirt Collar': 1,
                        'Peter Pan': 2,
                        'Puritan Collar': 3,
                        'Rib Collar': 4}
label2index[2] = collar_disign_labels

lapel_disign_labels = {'Invisible': 0,
                       'Notched': 1,
                       'Collarless': 2,
                       'Shawl Collar': 3,
                       'Plus Size Shawl': 4}
label2index[3] = lapel_disign_labels

neck_disign_labels = {'Invisible': 0,
                      'Turtle Neck': 1,
                      'Ruffle Semi-High Collar': 2,
                      'Low Turtle Neck': 3,
                      'Draped Collar': 4}
label2index[4] = neck_disign_labels

neckline_disign_labels = {'Invisible': 0,
                          'Strapless Neck': 1,
                          'Deep V Neckline': 2,
                          'Straight Neck': 3,
                          'V Neckline': 4,
                          'Square neckline': 5,
                          'Off Shoulder': 6,
                          'Round Neckline': 7,
                          'Sweat Heart Neck': 8,
                          'One Shoulder Neckline': 9}
label2index[5] = neckline_disign_labels

pant_length_labels = {'Invisible': 0,
                      'Short Pant': 1,
                      'Mid Length': 2,
                      '3/4 Length': 3,
                      'Cropped Pant': 4,
                      'Full Length': 5}
label2index[6] = pant_length_labels

sleeve_length_labels = {'Invisible': 0,
                        'Sleeveless': 1,
                        'Cup Sleeves': 2,
                        'Short Sleeves': 3,
                        'Elbow Sleeves': 4,
                        '3/4 Sleeves': 5,
                        'Wrist Length': 6,
                        'Long Sleeves': 7,
                        'Extra Long Sleeves': 8}
label2index[7] = sleeve_length_labels
