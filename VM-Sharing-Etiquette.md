
# Proper Etiquette for Sharing a Virtual Machine (VM)

Sharing a virtual machine with others, especially in a professional or collaborative environment, requires careful attention to etiquette. VMs often have limited resources, and collaboration on them can lead to confusion or inefficiency if best practices are not followed. This guide outlines the key points for VM-sharing etiquette and provides answers to common questions to ensure smooth cooperation.

## Table of Contents

1. [General Guidelines](#general-guidelines)
2. [Communication](#communication)
3. [Resource Management](#resource-management)
4. [File Organization](#file-organization)
5. [Security Best Practices](#security-best-practices)
6. [Environment Management](#environment-management)
7. [Frequent FAQs: Do's and Don'ts](#frequent-faqs-dos-and-donts)
8. [Example AI Training File Structure](#example-ai-training-file-structure)

---

## General Guidelines

### 1. Respect System Resources
- Always be mindful of CPU, memory, and disk usage.
- If your task is computationally heavy (e.g., training a large AI model), ensure others are aware and not affected by slowdowns or crashes.
- Don't run long, resource-intensive processes without notifying others.

### 2. Clean Up After Yourself
- Once you finish a task, ensure that you clean up unused files, shut down processes, and remove any environment-specific configurations that are no longer needed.
- Avoid leaving personal temporary files or unused VMs/data lying around, cluttering shared space.

### 3. Consistency in Naming
- Use clear, consistent naming conventions for directories, scripts, virtual environments, and files.
- Prefix your files or directories with your username or project name, e.g., `john_model_training/`.

---

## Communication

### 1. Notify Others of Your Activity
- If you're about to perform tasks that may take up significant resources, notify the team.
- Use a shared messaging platform (Slack, email, etc.) to communicate when you're starting, and when you've completed heavy tasks.

## Resource Management

### 1. Monitor System Load
- Regularly monitor CPU, memory, and disk usage using tools like `htop`, `nvidia-smi` (for GPU), or `df -h` for disk space.
- If resources are scarce, consult others before starting additional processes.

### 2. Use Virtual Environments
- Use virtual environments (venv, conda, etc.) for your Python projects or any language-specific environments.
- Ensure that everyone works in isolated environments to avoid dependency conflicts.

---

## File Organization

### 1. Follow Agreed-upon File Structures
- Agree on a common file structure for storing datasets, scripts, results, and logs.
- Keep your files well-organized by creating clear directories for each project or task.

### 2. Limit the Use of Home Directories
- Avoid using personal home directories (`/home/yourname`) to store shared project files. Instead, use project-specific directories located in common spaces like `/projects/` or `/data/`.

### 3. Delete Obsolete Files
- Regularly remove unused or obsolete files to avoid clutter. If unsure, communicate with the team before deleting shared files.

---
## Security Best Practices

### 1. Use SSH Keys
- If accessing the VM remotely, use secure methods like SSH keys rather than plain passwords.
- Avoid sharing your SSH key or credentials with others. If others need access, ensure they generate and use their own keys.

### 2. Monitor Permissions
- Be mindful of file permissions. Use `chmod` to set appropriate access permissions on your files.
- Don’t open your directories to everyone unless it's necessary (`chmod 777` should be avoided in most cases).

### 3. Update and Patch
- Regularly update your software to ensure it's patched against vulnerabilities. Coordinate these updates with the team.

---

## Environment Management

### 1. Use Virtual Environments (Venv)
- Create and activate virtual environments for each project to isolate dependencies. (Information on this is located in this repo but not in htis file)
- Always deactivate or remove environments that are no longer needed to free up resources.



### 2. Version Control
- Use version control tools like Git for your scripts and code, ensuring that collaborators have access to the latest versions.
- Set up a shared repository to manage and track changes.

---

## Frequent FAQs: Do's and Don'ts

### **Do**  
1. **Respect Usage Limits:** Always ask if you're unsure about resource consumption.
2. **Coordinate Jobs:** If you're about to run a big job, communicate with the team to avoid bottlenecks.
3. **Use Virtual Environments:** They prevent conflicts between different projects.
4. **Backup Important Data:** Use external drives or cloud storage to back up important results. (locally on your computer will be fine, We will explain scp when needed)
5. **Monitor Your Jobs:** Keep an eye on your jobs, especially long-running tasks. Don’t let jobs run unattended without oversight.

### **Don’t**
1. **Don’t Leave Processes Unattended**: Ensure that long-running tasks won’t affect other users negatively by leaving them without monitoring.
2. **Don’t Hard Code Paths:** Avoid hard-coding paths that are specific to your user space. Instead, use relative paths or variables that generalize the environment for others.
3. **Don’t Store Unnecessary Files:** Clean up your environment by removing unnecessary files and directories after completion of your tasks.
4. **Don’t Override Shared Configurations:** Avoid modifying shared configuration files or system settings without first consulting with other users.

---
## Example AI Training File Structure

This is an example of how to set up the Linux file structure for AI training, ensuring separation between users and proper environment management.

```plaintext
/
├── home/
│   ├── user1/
│   │   ├── projects/
│   │   │   ├── ai_project_1/
│   │   │   │   ├── venv/                 # Virtual environment specific to this project
│   │   │   │   ├── data/                 # Raw data and datasets
│   │   │   │   ├── scripts/              # Python or shell scripts for training
│   │   │   │   ├── logs/                 # Log files from training processes
│   │   │   │   ├── models/               # Trained models or checkpoints
│   │   │   │   └── results/              # Results (accuracy, loss, etc.)
│   │   └── shared_scripts/               # Commonly used scripts across projects
│   ├── user2/
│   │   ├── projects/
│   │   │   └── ai_project_2/
│   │   │       ├── venv/
│   │   │       ├── data/
│   │   │       ├── scripts/
│   │   │       ├── logs/
│   │   │       ├── models/
│   │   │       └── results/
├── shared_data/
│   ├── datasets/                          # Datasets accessible by all users
│   └── preprocessed/                      # Preprocessed data shared by users
├── projects/
│   ├── ai_project_3/                      # Shared project worked on by multiple users
│   │   ├── user1_env/                     # User 1's specific environment
│   │   ├── user2_env/                     # User 2's specific environment
│   │   ├── data/
│   │   ├── scripts/
│   │   ├── logs/
│   │   ├── models/
│   │   └── results/
└── tools/
    ├── conda/                             # Conda installations
    └── other_ml_tools/                    # Shared tools for AI development (e.g., TensorFlow, PyTorch)
```

### Key Points:
- Each user has their own home directory where their projects reside.
- Shared data and tools are placed in `/shared_data/` and `/tools/` for collaboration.
- The `venv/` directory is specific to each project, ensuring no cross-contamination of dependencies.
- The directory structure is clean and allows for easy navigation across individual and shared projects.
