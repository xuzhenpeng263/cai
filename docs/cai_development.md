Development is facilitated via VS Code dev. environments. To try out our development environment, clone the repository, open VS Code and enter de dev. container mode:

![CAI Development Environment](media/cai_devenv.gif)


### Contributions

If you’d like to contribute to this project, feel free to jump in! Whether it’s fixing bugs, improving documentation, or adding new features—your help is welcome and appreciated. Please open an issue or submit a pull request to get started.

### Usage Data Collection

CAI is provided free of charge for researchers. Instead of payment for research use cases, we ask you to contribute to the CAI community by allowing usage data collection. This data helps us understand how the framework is being used, identify areas for improvement, and prioritize new features. The collected data includes:

- Basic system information (OS type, Python version)
- Username and IP information
- Tool usage patterns and performance metrics
- Model interactions and token usage statistics

We take your privacy seriously and only collect what's needed to make CAI better.

### Reproduce CI-Setup locally

To simulate the CI/CD pipeline, you can run the following in the Gitlab runner machines:

```bash
docker run --rm -it \
  --privileged \
  --network=exploitflow_net \
  --add-host="host.docker.internal:host-gateway" \
  -v /cache:/cache \
  -v /var/run/docker.sock:/var/run/docker.sock:rw \
  registry.gitlab.com/aliasrobotics/alias_research/cai:latest bash
```