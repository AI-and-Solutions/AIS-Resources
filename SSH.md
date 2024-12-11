Full Guide: Setting Up and Using Your PC as a Virtual Machine (VM)

This guide provides an in-depth walkthrough of setting up a physical PC as a virtual machine host and configuring it for remote development and collaboration. The process includes installing Ubuntu, configuring the network, securing remote access, setting up development tools, and ensuring seamless team collaboration.

### 1. Install Ubuntu on a Partition

## Step 1.1: Preparing for Installation

Download Ubuntu ISO: Obtain the latest Ubuntu LTS version from ubuntu.com.

# Create a Bootable USB:

Use tools like Rufus (Windows), Balena Etcher (cross-platform), or dd (Linux) to create a bootable USB.

Ensure the USB is at least 8 GB.

Validate the integrity of the ISO file using checksums (e.g., sha256sum).

## Step 1.2: Partitioning and Installation

Partitioning:

Before booting into the USB, back up all important data.

Use tools like GParted to create a 250 GB ext4 partition for Ubuntu.

Allocate swap space (~2x the size of RAM) if needed.

# Installing Ubuntu:

Boot into the USB by adjusting BIOS/UEFI boot order.

Select “Install Ubuntu” and choose “Manual Partitioning”.

Assign the ext4 partition as root (/) and configure swap space if applicable.

## Step 1.3: Post-Installation Configuration

# First Boot:

Log in using the primary user credentials created during installation.

Update the system:

sudo apt update && sudo apt upgrade -y

# Essential Tools:

## Install basic development packages:

sudo apt install git build-essential python3-pip curl

# Install OpenSSH for remote access:

sudo apt install openssh-server

# Verify OpenSSH is running:

sudo systemctl status ssh

### 2. Configure the Network

## Step 2.1: Setting a Static IP Address

# Determine Network Interface:

ip a

Identify the active interface (e.g., eth0 or wlan0).

Edit Netplan Configuration:

sudo nano /etc/netplan/01-netcfg.yaml

# Example configuration:

network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# Apply Changes:

sudo netplan apply

# Verify Configuration:

ping google.com

## Step 2.2: Hostname Configuration

# Set a Descriptive Hostname:

sudo hostnamectl set-hostname UbuntuAIHost

Update /etc/hosts:

sudo nano /etc/hosts

# Add:

192.168.1.100 UbuntuAIHost

## Step 2.3: Router Configuration

Port Forwarding:

Access your router settings (typically at 192.168.1.1).

Forward port 22 (or a custom SSH port) to your Ubuntu system’s IP.

### 3. Enable and Secure the SSH Server

## Step 3.1: Enable and Start SSH

# Enable SSH on Boot:

sudo systemctl enable ssh

# Verify Service Status:

sudo systemctl status ssh

## Step 3.2: Enhance Security

# Modify SSH Configuration:

sudo nano /etc/ssh/sshd_config

# Key changes:

Change default port:

Port 2222

Disable root login:

PermitRootLogin no

Enable public key authentication:

PasswordAuthentication no

Restart SSH Service:

sudo systemctl restart ssh

## Step 3.3: Set Up Public Key Authentication

Generate SSH Keys:

On each client machine:

ssh-keygen -t rsa -b 4096 -f ~/.ssh/ubuntu_key

# Copy Public Keys to Server:

ssh-copy-id -i ~/.ssh/ubuntu_key.pub -p 2222 username@192.168.1.100

### 4. Generate and Distribute PEM Files

## Step 4.1: Generate Key Pairs

# Server-Side Key Generation:

ssh-keygen -t rsa -b 2048 -f ~/.ssh/teammate_key

## Step 4.2: Securely Share PEM Files

# Distribute Keys:

Use tools like scp or secure email.

# Example command:

scp -P 2222 ~/.ssh/teammate_key.pem teammate@remote_host:/path/to/store

# Protect Keys:

Ensure PEM file permissions:

chmod 400 teammate_key.pem

### 5. Configure Firewall Settings

## Step 5.1: Enable UFW

# Basic Rules:

sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 2222/tcp
sudo ufw enable

## Step 5.2: Monitor Firewall Logs

# Check UFW Status:

sudo ufw status verbose

# Review Logs:

sudo journalctl -u ufw
