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

