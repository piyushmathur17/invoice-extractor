from os import listdir,mkdir
from pdf2image import convert_from_path

dir_path = "/home/piyush/Pictures/invoices/"
destination = "/home/piyush/mygit/invoice-extractor/invoice_images/"

pdfs = listdir(dir_path)
#print(pdfs)
for pdf in pdfs:
	file = dir_path + pdf
	dir_name = pdf.split('.')[0]
	target = destination + dir_name
	#mkdir(target)
	pages = convert_from_path(file)
	j=1
	for page in pages:
		page.save(target+'/'+str(j)+'.jpg', 'JPEG')
		j+=1