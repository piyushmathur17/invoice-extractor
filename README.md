# invoice-extractor
A python implementation to extract data in structured form from an image of an invoice

# Flow:
## original invoice 
![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/sample_output_original.png)

## preprocessing

### removing lines
this is being done to accurately detect text contours

### mask obtained for vertical and horizontal lines
![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/X_lines.jpg)

![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/Y_lines.jpg)

### after applying mask
![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/lines_removed.jpg)

## Obtained graph
![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/graph.jpg)
after getting contours and merging them on the basis of their size and nearness
*the red boxes are the identified keyfields
 the keyfields can be changes according to keywords given in labels.csv and label_synonnyms.csv
*green boxes are the values
*relation between the keyfields and it's possible values is shown by using straight lines

# Output csv
![alt text](https://github.com/piyushmathur17/invoice-extractor/blob/master/invoice_images/output.png)

