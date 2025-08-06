# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email**: Send details to [your-email@domain.com] with "SECURITY" in the subject line
2. **Include**: Detailed description of the vulnerability
3. **Provide**: Steps to reproduce the issue
4. **Attach**: Any relevant code or screenshots

### 📝 What to Include

Please provide as much information as possible:

- **Type of vulnerability** (e.g., data exposure, injection, authentication bypass)
- **Location** (file path, line number, function name)
- **Impact assessment** (what data could be accessed/modified)
- **Reproduction steps** with specific details
- **Suggested fix** if you have one

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Vulnerability Assessment**: Within 1 week
- **Fix Development**: Depends on severity (1-4 weeks)
- **Public Disclosure**: After fix is released

### 🛡️ Security Considerations

This application handles sensitive financial data. Key security areas:

#### Email Credentials
- Store email passwords in environment variables
- Use app passwords instead of main account passwords
- Rotate credentials regularly

#### Portfolio Data
- Portfolio holdings and values are sensitive
- Ensure .env file is never committed to version control
- Consider encrypting stored portfolio data

#### API Keys
- Secure storage of financial data API keys
- Implement rate limiting to prevent abuse
- Validate all external API responses

#### Data Transmission
- All email communications should use TLS/SSL
- Secure API connections (HTTPS only)
- Validate and sanitize all input data

### 🔍 Known Security Considerations

#### Email Configuration
- Uses SMTP with TLS encryption
- Credentials stored in environment variables
- Email content contains portfolio performance data

#### External Dependencies
- Regular updates of yfinance and other packages
- Monitor for security advisories in dependencies
- Validate external data sources

#### Data Logging
- Avoid logging sensitive portfolio values
- Sanitize error messages before logging
- Secure log file permissions

### 🚨 Security Best Practices

#### For Users
1. Use strong, unique passwords for email accounts
2. Enable 2FA on email accounts
3. Regularly rotate API keys and passwords
4. Keep the application updated
5. Run the application in a secure environment

#### For Developers
1. Follow secure coding practices
2. Validate all inputs from external sources
3. Use parameterized queries if adding database support
4. Implement proper error handling without information disclosure
5. Regular security code reviews

### 📊 Vulnerability Severity Classification

| Severity | Description | Examples |
|----------|-------------|----------|
| **Critical** | Immediate threat to data security | Remote code execution, credential theft |
| **High** | Significant security impact | Data exposure, authentication bypass |
| **Medium** | Moderate security impact | Information disclosure, DoS vulnerabilities |
| **Low** | Minor security impact | Timing attacks, minor information leaks |

### 🏆 Recognition

We appreciate security researchers who help us maintain a secure application:

- Public acknowledgment (with permission)
- Credit in release notes
- Priority support for future security reports

### 📞 Contact Information

For security-related concerns:
- **Email**: [your-email@domain.com]
- **Subject**: SECURITY - [Brief Description]
- **Encryption**: PGP key available upon request

### 🔄 Security Update Process

1. **Assessment**: Evaluate reported vulnerability
2. **Development**: Create and test security fix
3. **Testing**: Comprehensive security testing
4. **Release**: Deploy fix with security advisory
5. **Notification**: Inform users of security update

### 📋 Security Checklist for Contributors

Before submitting code:

- [ ] No hardcoded credentials or API keys
- [ ] Input validation for all external data
- [ ] Proper error handling without information disclosure
- [ ] Dependencies are up to date
- [ ] No sensitive data in logs
- [ ] Secure configuration examples

### 🔐 Encryption and Data Protection

#### Data at Rest
- Environment variables for sensitive configuration
- Consider encrypting portfolio data files
- Secure file permissions on configuration files

#### Data in Transit
- SMTP TLS encryption for email reports
- HTTPS for all API communications
- Validate SSL certificates

### ⚠️ Disclaimer

This security policy is provided as-is. Users are responsible for:
- Securing their own environment and credentials
- Keeping the application and dependencies updated
- Following security best practices
- Regular security assessments of their deployment

The maintainers make no warranties about the security of this software and are not liable for any security incidents or data breaches.

---

**Security is a shared responsibility. Thank you for helping keep our community safe! 🛡️**
