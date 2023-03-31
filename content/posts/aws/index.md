---
title: "A beginner’s guide to running Jupyter Notebook on Amazon EC2"
date: 2019-06-03
draft: false
---

As a beginner in large-scale data manipulation, I quickly found the computational needs of my projects exceeding the capabilities of my personal equipment. I have therefore found Amazon’s EC2 offering very convenient -- renting virtual computers on which computer applications can be run remotely from a local machine, and for free.

In this post, I detail the process that I have followed to set up an EC2 instance.

##### Overview:
1. Create an AWS account
2. Launch an EC2 instance via the EC2 dashboard
3. Connect to your EC2 instance using SSH
4. Install Anaconda to your EC2 instance
5. Configure Jupyter Notebook
6. Connect to Jupyter Notebook from your local machine
7. Stop your EC2 instance

##### Step 1: Create an AWS account
- Create an AWS account [here](https://aws.amazon.com/).
- AWS treats **new users with free tier benefits for 12 months**, which will automatically be applied to your account. In the sign-up form, you’ll have to provide your card details, but you’ll only be charged if you exceed the generous free tier limits. **EC2 has limits on both the type of instance that you can use** (t2.micro) and **how many hours you can use in one month** (750 hours). Unfortunately, EC2 instances with GPUs are not included in the free tier, but the cost can be low, around $0.65–$0.9/h. You can find additional details about the free tier [here](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all).
- In order to keep a peace of mind, set up billing reminders to get notified when your AWS service usage is approaching, or has just exceeded, the free tier usage limits.

##### Step 2: Create an EC2 instance via the EC2 Dashboard
- To access your EC2 Dashboard, sign in to your [AWS Management console](https://aws.amazon.com/console/), click on *Services* on the top and then *EC2*.
- Click on *Instances* under the *Instances* menu on the left hand side to visit your instances board. This is where you can create, monitor and control your EC2 instances.
- To create a new instance, click on *Launch Instance*.

{{< figure src="images/blog_aws_0.png" width="90%">}}

- In Step 1, you need to select a *pre-configured Amazon Machine Image (AMI), which provides the information needed to launch your virtual server in the cloud*.The key distinguishing factors are free tier eligibility, type of OS and pre-settings that you want installed in your machine. As I didn’t have any specific requirements, I selected a basic Ubuntu server (*Ubuntu Server 18.04 LTS (HVM), SSD Volume Type*)

{{< figure src="images/blog_aws_1.png" width="80%">}}

- In Step 2, you need to choose the type of instance that meets your CPU, memory, storage, and networking capacity requirements. However, only the t2.micro instance is eligible for the free tier.

{{< figure src="images/blog_aws_2.png" width="80%">}}

- Accept the default settings in steps 3–5.
- In Step 6, you need to provide the *TCP protocol port 8888*, to allow the Jupyter Notebook to launch from a browser. Click on Add Rule and fill in the details for a Custom TCP Rule as shown below.

{{< figure src="images/blog_aws_3.png" width="80%">}}

- In Step 7, review your settings and press Launch. You’ll be prompted to create a key pair, which contains the necessary keys for encrypting and decrypting your login information. As you’ll use the key pair for authentication everytime you connect to EC2, it is wise to *use a simple name and save it to an easily accessible location*. Click on *Download Key Pair* and then Launch Instances.

{{< figure src="images/blog_aws_4.png" width="80%">}}

- If your key pair is downloaded as a .txt file, simply change the extension to .pem .

{{< figure src="images/blog_aws_5.png" width="80%">}}

- Back on your browser, click on *View Instances* to return to your instances board. You should be able to see your instance being initiated, which may take a few minutes. Once this is done, the Instance State will turn to running.

{{< figure src="images/blog_aws_6.png" width="80%">}}


##### Step 3: Connect to your EC2 instance using SSH
- Select your instance on the EC2 dashboard and press Connect. This will open a window with instructions that we’ll follow to access EC2 programmatically.
- Open a bash terminal and use the cd command to change your working directory to the location where your .pem file is located. Use *ls* to check the contents of your working directory.

{{< figure src="images/blog_aws_7.png" width="90%">}}

- If it’s the first time that you use your pem file, run *sudo chmod 400 ?.pem* to make it private, where ? is your .pem file name. Then, enter your computer’s user password. Run this command again in the future if you get the error *WARNING: UNPROTECTED PRIVATE KEY FILE!*

{{< figure src="images/blog_aws_8.png" width="90%">}}

- Next, SSH into your EC2 instance using *ssh -i ?.pem ubuntu@??* where ? is your pem file name and ?? is your Public Domain Name System (DNS). You can find it in the EC2 Instance dashboard under the Public DNS column, or in the instructions window for connecting to your instance as shown below.

{{< figure src="images/blog_aws_9.png" width="80%">}}
{{< figure src="images/blog_aws_10.png" width="80%">}}

- Once you execute the SSH command, you’ll be prompted with a yes/no question. Type yes and you should be SSH-ed into your instance.

{{< figure src="images/blog_aws_11.png" width="80%">}}

You’re now running a virtual machine in the cloud!

##### Step 4: Install Anaconda to your EC2 instance
- Visit [Anaconda’s download page](https://www.anaconda.com/distribution/#download-section) to get the url of the latest version of the Linux 64-bit version. In my case the url was: https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh .

{{< figure src="images/blog_aws_12.png" width="80%">}}

- Run the *wget* command followed by the url to download Anaconda.

{{< figure src="images/blog_aws_13.png" width="80%">}}

- Install Anaconda by running *bash Anaconda3-2018.12-Linux-x86_64.sh* or *bash* followed by the file name indicated next to ‘Saving to:’ in your terminal.
- Press enter several times to get through all the legalese, and yes to agree with the licence terms.
- At the end you should be prompted to include Anaconda3 into your .bashrc PATH. If this doesn’t happen or you hit enter before doing so, you’ll have to manually enter the PATH in your .bashrc file. Run nano .bashrc to access it. Nano is my preferred command line text editor, as it’s pseudo-graphical layout makes it much more user friendly.
- Scroll to the bottom of the file, and insert *export PATH=/home/ubuntu/anaconda3/bin:$PATH* at the very end of the file. Press control and X to exit.

{{< figure src="images/blog_aws_14.png" width="80%">}}


- Then type Y to save

{{< figure src="images/blog_aws_15.png" width="80%">}}

- Finally press *Enter* to confirm the file name to be updated, which should take you back to the terminal view.

{{< figure src="images/blog_aws_16.png" width="80%">}}

- When you edit your bashrc file you need to log out and back in to make sure that your changes take effect. Thankfully the *source .bashrc* command can take care of that.
You’ll also have to check whether Anaconda3 is your default Python environment, by running *which python /home/ubuntu/anaconda3/bin/python*. Below is the expected behaviour:

{{< figure src="images/blog_aws_17.png" width="80%">}}

##### Step 5: Configure Jupyter Notebook
- Type ipython to run python 3
- Then, type *from IPython.lib import passwd* and *passwd()* to set a password for your Jupyter Notebook, in order to prevent any unauthorized access.
- You’ll be prompted to provide a password. Once verified, make sure to note your password and hashed password (SHA-1) as they’ll be needed in later steps. Run *exit* or press control and d to terminate python.

{{< figure src="images/blog_aws_18.png" width="80%">}}

- Next, you’ll configure Jupyter Notebook to access your notebooks from your local computer via an internet browser. First, create a configuration file by typing *jupiter notebook --generate-config*

{{< figure src="images/blog_aws_19.png" width="80%">}}

- When using a password, it is a good idea to also use SSL with a web certificate, so that your hashed password is not sent unencrypted by your browser.
- Generate SSL certificates so that your browser trusts the Jupyter Notebook server, by typing mkdir certs . Then, run cd certs to go into your certs directory.
- Create a new PEM file (I named mine mycert.pem), by typing *sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem*. You’ll be asked to provide some personal information for your certificate as shown below.

{{< figure src="images/blog_aws_20.png" width="90%">}}

- Once you’re done, change permission to your .pem file by running sudo chown $USER:$USER mycert.pem .

{{< figure src="images/blog_aws_21.png" width="90%">}}

- Go back to your home directory by running *cd*.
- Now, you’ll edit your Jupyter configuration file. Type *cd ~/.jupyter/* and then *nano jupyter_notebook_config.py*

{{< figure src="images/blog_aws_22.png" width="90%">}}

- Paste the below text after adjusting it to reflect your instance details at the top of the file. When you’re done, press control and X to exit, then type Y to save and finally press Enter.

    ```python
    c = get_config()

    # Kernel config
    c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook

    # Notebook config
    c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem'
    #location of your certificate file
    c.NotebookApp.ip = '0.0.0.0'
    c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
    # ***Edit this with the SHA hash that you generated earlier***
    c.NotebookApp.password = u'sha1:....enter your hash here....'
    # This is the port you opened in Step 1 when launching an EC2 instance
    c.NotebookApp.port = 8888
    ```

{{< figure src="images/blog_aws_23.png" width="75%">}}

- Go back to your home directory by typing *cd ~*.
- Create a folder called Notebook by typing mkdir Notebooks and change your working directory to that folder by typing *cd Notebooks*.

##### Step 6: Connect to Jupyter Notebook from your local machine
- Type jupyter notebook to run Jupyter Notebook on your EC2 instance.

{{< figure src="images/blog_aws_24.png" width="75%">}}

- Finally, access Jupyter Notebook from your browser using your Public DNS that we discussed in Step 3. Make sure to add “https://” before and “:8888” at the end. Mine looks like:
https://ec2-34-244-48-16.eu-west-1.computeamazonaws.com:8888

- This link will take you to a warning screen:

{{< figure src="images/blog_aws_25.png" width="65%">}}

- Click on *Show Details* in Safari or Advanced in Chrome, and then *visit this website*.

{{< figure src="images/blog_aws_26.png" width="65%">}}

- Your browser may have also warned you that your certificate is invalid or insecure. If you wish to create a fully compliant certificate that will not raise warnings, it is possible (but rather elaborate) to create one, as explained [here](https://arstechnica.com/information-technology/2009/12/how-to-get-set-with-a-secure-sertificate-for-free/).
- Next, type in the password that you chose in Step 5 (not the SHA hash).

{{< figure src="images/blog_aws_27.png" width="65%">}}

and you’re done!

##### Step 7: Stop EC2
- Your Jupyter Notebook server **will keep running until you deliberately stop it, or you stop the server** (closing your laptop is fine!).
- When you’re done working on your EC2 instance **you should stop it to prevent being charged for time not using it**. To do this, go to your instances dashboard, right click on your instance, then click instance state and *stop*.
- Next time that you want to access your instance, right click on it, then press *instance state* and then *start*.
- You can then *ssh* into your instance as we did in Step 2. Note that the public and private IP of an EC2 instance does not persist through stops/starts so you’ll have to use an updated Public DNS, as shown in your instances dashboard under the Public DNS column or the instructions window that appears when you connect to your instance.
