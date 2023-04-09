php-plots
=========

PHP based web index for image displaying.

Original code from Giovanni Petrucciani (@gpetruc).

This project contains a PHP web index script to help visualizing folders with many images.

It is tailored to people working in High Energy Physics that use ROOT (http://root.cern.ch) to produce their plots.

# Setup

1. cd into your web folder

        cd <my-path>
        
1. Clone this repository (and its submodule jsroot)

        git clone --recursive git@github.com:terrill37/php-plots.git 
        
1. Copy the .htaccess file into .htaccess on the webpage directory and edit its content to suit your needs.
 
    a. .htaccess restricts webpage access to valid users from group
    
    b. edit .htaccess for your restrictions 
    
    c. .htaccess is not required for webpage to work
        
        cp -p php-plots/.htaccess .htaccess
        $EDITOR .htaccess

1. Copy res/  jsroot/ and index.php into your web page top directory.

        cp -r res/ /path/to/www/
        cp -r jsroot/ /path/to/www/
        cp index.php /path/to/www/
        
1. add plots in .jpeg, .png, .jpg format to webpage top directory

    a. make sure pdfs have same name as image to be associated with

1. Open the web folder into your browser.

1. Enjoy.


# Features

1. Detect if a file is present with multiple formats.

1. Filter files to be selected with wild-cards or regex.

1. Zoom in/out images with double-click.

1. Rearrange images with drag and drop.

1. Overlay content of .txt version on mouse hover.

# Example

https://musella.web.cern.ch/musella/php-plots
