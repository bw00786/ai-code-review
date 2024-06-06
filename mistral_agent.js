const { warning, info } = require("@actions/core");
const { MistralAI } = require('@xenova/transformers'); // Assuming 'mistralai' is the library for Mistral's API

class MistralAPI {

    constructor(apiKey, fileContentGetter, fileCommentator) {
        try {
            this.mistral = new MistralAI({ apiKey });
        } catch (error) {
            throw new Error("Failed to initialize Mistral AI client: " + error.message);
        }
        this.fileContentGetter = fileContentGetter;
        this.fileCommentator = fileCommentator;
        this.fileCache = {};
    }

    async initCodeReviewAssistant() {
        try {
            this.assistant = await this.mistral.assistants.create({
                name: "Mistral-7B AI core-reviewer",
                instructions:
                    "You are the smartest Mistral-7B AI responsible for reviewing code in our company's GitHub PRs.\n" +
                    "Review the user's changes for logical errors and typos.\n" +
                    "- Use the 'addReviewCommentToFileLine' tool to add a note to a code snippet containing a mistake. Pay extra attention to line numbers.\n" +
                    "Avoid repeating the same issue multiple times! Instead, look for other serious mistakes.\n" +
                    "And a most important point - comment only if you are 100% sure! Omit possible compilation errors.\n" +
                    "- Use 'getFileContent' if you need more context to verify the provided changes!\n" +
                    "Warning! Lines in any file are calculated from 1. You should complete your work and provide results to the user only via functions!",
                model: "Mistral-7B-Instruct-v0.3",
                tools: [{
                    "type": "function",
                    "function": {
                        "name": "getFileContent",
                        "description": "Retrieves the file content to better understand the provided changes",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pathToFile": {
                                    "type": "string",
                                    "description": "The fully qualified path to the file."
                                },
                                "startLineNumber": {
                                    "type": "integer",
                                    "description": "The starting line number of the code segment of interest."
                                },
                                "endLineNumber": {
                                    "type": "integer",
                                    "description": "The ending line number of the code segment of interest."
                                }
                            },
                            "required": ["pathToFile", "startLineNumber", "endLineNumber"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "addReviewCommentToFileLine",
                        "description": "Adds an AI-generated review comment to the specified line in a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fileName": {
                                    "type": "string",
                                    "description": "The relative path to the file."
                                },
                                "lineNumber": {
                                    "type": "integer",
                                    "description": "The line number in the file where the issue was found."
                                },
                                "foundIssueDescription": {
                                    "type": "string",
                                    "description": "Description of the issue found."
                                }
                            },
                            "required": ["fileName", "lineNumber", "foundIssueDescription"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "codeReviewDone",
                        "description": "Marks the code review as completed.",
                        "parameters": {}
                    }
                }]
            });
        } catch (error) {
            warning("Failed to initialize Mistral assistant: " + error.message);
            throw error;
        }
    }

    async getFileContent(args) {
        const { pathToFile, startLineNumber, endLineNumber } = args;
        const span = 20;

        let content = '';
        try {
            if (pathToFile in this.fileCache) {
                content = this.fileCache[pathToFile];
            } else {
                content = await this.fileContentGetter(pathToFile);
                this.fileCache[pathToFile] = content;
            }
        } catch (error) {
            throw new Error("Failed to retrieve file content: " + error.message);
        }

        const start = Math.max(startLineNumber - span, 0);
        const end = Math.min(endLineNumber + span, content.length);
        return `${pathToFile}\n'''\n${content.substring(start, end)}\n'''\n`;
    }

    async addReviewCommentToFileLine(args) {
        const { fileName, lineNumber, foundIssueDescription } = args;
        try {
            await this.fileCommentator(foundIssueDescription, fileName, lineNumber);
            return "The note has been published.";
        } catch (error) {
            return `There is an error in the 'addReviewCommentToFileLine' usage! Error message:\n${JSON.stringify(error)}`;
        }
    }

    async doReview(changedFiles) {
        const simpleChangedFiles = changedFiles.map(file => ({
            filename: file.filename,
            status: file.status,
            additions: file.additions,
            deletions: file.deletions,
            changes: file.changes,
            patch: file.patch
        }));

        await this.initCodeReviewAssistant();

        let retries = 0;
        const maxRetries = 3;
        while (retries < maxRetries) {
            try {
                this.thread = await this.mistral.threads.create();
                await this.doReviewImpl(simpleChangedFiles);
                break;
            } catch (error) {
                await this.mistral.threads.del(this.thread.id);
                warning(`Error encountered: ${error.message}; retrying...`);
                retries++;
                if (retries >= maxRetries) {
                    warning("Max retries reached. Unable to complete code review.");
                    throw error;
                }
                const delay = Math.pow(2, retries) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    async doReviewImpl(simpleChangedFiles) {
        try {
            this.message = await this.mistral.threads.messages.create(
                this.thread.id,
                {
                    role: "user",
                    content: JSON.stringify(simpleChangedFiles)
                }
            );

            this.run = await this.mistral.threads.runs.createAndPoll(
                this.thread.id,
                {
                    assistant_id: this.assistant.id,
                }
            );

            await this.processRun();

            const messages = await this.mistral.threads.messages.list(this.thread.id);

            for (const message of messages.data.reverse()) {
                console.log(`${message.role} > ${message.content[0].text.value}`);
            }
        } catch (error) {
            warning("Error during doReviewImpl: " + error.message);
            throw error;
        }
    }

    async processRun() {
        do {
            this.runStatus = await this.mistral.threads.runs.retrieve(this.thread.id, this.run.id);

            if (this.runStatus.status === 'requires_action') {
                const tools_results = [];
                for (const toolCall of this.runStatus.required_action.submit_tool_outputs.tool_calls) {
                    let result = '';
                    const args = JSON.parse(toolCall.function.arguments);
                    try {
                        if (toolCall.function.name === 'getFileContent') {
                            result = await this.getFileContent(args);
                        } else if (toolCall.function.name === 'addReviewCommentToFileLine') {
                            result = await this.addReviewCommentToFileLine(args);
                        } else if (toolCall.function.name === 'codeReviewDone') {
                            return;
                        } else {
                            result = `Unknown tool requested: ${toolCall.function.name}`;
                        }
                    } catch (error) {
                        result = `Error processing ${toolCall.function.name}: ${error.message}`;
                    }
                    tools_results.push({ tool_call_id: toolCall.id, output: result });
                }

                await this.mistral.threads.runs.submitToolOutputs(this.thread.id, this.run.id, {
                    tool_outputs: tools_results,
                });
            }

            await new Promise(resolve => setTimeout(resolve, 1000));
        } while (this.runStatus.status !== "completed");
    }
}

module.exports = MistralAPI;

