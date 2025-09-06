summarization = {}
translation = {}

summarization_java_1 = """# Step 1: The function is named 'check' and takes two parameters: a String 'passwd' and a String 'hashed'.
# Step 2: The return type is boolean, so the function returns true or false depending on some condition.
# Step 3: The body of the function parses the hashed string, extracts scrypt parameters and salt, and recomputes the derived key.
# Step 4: It compares the recomputed key with the original using a constant-time check.
# Step 5: The function's purpose is to verify password correctness based on scrypt hashing.
# Step 6: Since the function includes parameters and a return value, '@param' and '@return' tags should be used to help users understand the inputs and outputs.
# Step 7: Because the user requests a brief summary, the final summary will be a short sentence, plus param and return tags."""

summarization_java_2 = """# Step 1: The function is named 'detect' and takes no input parameters.
# Step 2: The return type is 'Platform', so the function returns a Platform object.
# Step 3: It retrieves system architecture and operating system name using system properties.
# Step 4: It iterates through predefined architecture and OS enums to find matches using regex.
# Step 5: If a matching pair is found, a new Platform object is returned. If no match is found, the function throws an UnsupportedPlatformException.
# Step 6: The function attempts to detect the current execution platform based on system info.
# Step 7: Since it has a return value and may throw an exception, '@return' and '@throws' tags are appropriate for the summary.
# Step 8: The final summary will be a brief and concice sentence, plus tags @return and @throws."""

summarization_java_3 = """# Step 1: The function is named 'getNodeMetaData' and takes one parameter: 'key'.
# Step 2: It has a generic return type '<T>', meaning it returns a value of type T.
# Step 3: It checks whether the internal map 'metaDataMap' is null.
# Step 4: If null, it returns null casted to type T.
# Step 5: Otherwise, it retrieves the value associated with 'key' from the map and casts it to type T.
# Step 6: The function's purpose is to retrieve metadata for a given key from the node's metadata map.
# Step 8: The final summary will be a brief and concice sentence."""


# ------------------------------Python----------------------------
summarization_python_1 = """# Step 1: The function is named 'settext' and is defined as a method of a class (it takes 'self' as its first parameter).
# Step 2: It accepts two parameters: 'text' (required) and 'cls' (optional, with default value 'current').
# Step 3: The docstring explains the purpose and arguments of the function.
# Step 4: 'text' is expected to be a string representing the content to set.
# Step 5: 'cls' is also a string that specifies the class of the text, and its default behavior is described in the docstring.
# Step 6: The body of the function calls 'self.replace' with the class 'TextContent' and keyword arguments 'value=text' and 'cls=cls'.
# Step 7: This indicates that the method updates or replaces the text content associated with this object, categorized by 'cls'.
# Step 8: The final summary will be a brief and concice sentence."""

summarization_python_2 = """# Step 1: The function is named 'setdocument' and takes one parameter: 'doc'.
# Step 2: It asserts that 'doc' is an instance of the 'Document' class.
# Step 3: If 'self.doc' is not already set, it assigns 'doc' to it.
# Step 4: If the element has an ID, it checks whether the ID already exists in 'doc'; if so, it raises 'DuplicateIDError'.
# Step 5: If the ID is not present, the element is added to the document’s index.
# Step 6: The method then iterates over all children and recursively sets the document for those that are instances of 'AbstractElement'.
# Step 7: The function ensures that the element and all its descendants are properly linked to the same document with unique IDs.
# Step 8: The final summary will be a brief and concise sentence."""

summarization_python_3 = """# Step 1: The function is named 'addable' and checks if an element of the given class can be added to a parent element.
# Step 2: It first verifies whether the parent accepts the class by calling 'parent.__class__.accepts'.
# Step 3: If the class has a positive 'OCCURRENCES' limit, it checks whether the parent already has too many instances.
# Step 4: If the limit is reached, it raises a 'DuplicateAnnotationError' or returns False based on 'raiseexceptions'.
# Step 5: If 'OCCURRENCES_PER_SET' is specified and a 'set' is provided, it checks set-level constraints similarly.
# Step 6: In both checks, 'parent.count()' is used to determine how many existing elements of the class are present.
# Step 7: The function returns True only if all constraints are satisfied, meaning the element is addable.
# Step 8: The final summary will be a brief and concise sentence."""

summarization_go_1 = """#Step 1: The function accepts a rune slice and returns an integer plus an error, which suggests it processes input text and reports how much was consumed.
#Step 2: It immediately checks if the first rune is a double quote; if not, it fails, so the input must represent a quoted string.
#Step 3: It then scans forward, tracking characters until it finds a closing quote that is not escaped. This ensures the string ends properly.
#Step 4: The code includes logic for escape sequences, though commented out, showing that the parser is designed to handle escaped characters within the string.
#Step 5: If no valid closing quote is discovered, an error is returned indicating the string is malformed.
#Step 6: Otherwise, the function returns the number of runes consumed, marking the span of the quoted string.
#Step 7: From these behaviors, the concise meaning is: the function parses a quoted string from input, returns how many characters were read, and produces an error if the string is improperly formatted. """

summarization_go_2 = """#Step 1: The function takes a rune slice and returns an integer plus an error, so again it is parsing some textual input.
#Step 2: It first checks whether the slice is at least four runes long; if not, it immediately fails, which makes sense since valid boolean literals like “true” or “false” need at least four characters.
#Step 3: It then loops through a set of predefined literalValues, which presumably include valid boolean representations such as “true” and “false.”
#Step 4: For each candidate literal, it ensures the input is long enough and then calls isLitValue to check if the literal matches the beginning of the input. If a match is found, it records the length of that literal.
#Step 5: After scanning, if no match was found (n == 0), the function returns an error saying the boolean value is invalid.
#Step 6: Otherwise, it returns the length of the matched literal, representing how many characters were consumed.
#Step 7: From this logic, the distilled meaning is: the function parses a boolean value, returns the number of bytes read, and produces an error if the input is not a valid boolean."""

summarization_go_3 = """#Step 1: The function takes a rune slice and returns three values: an integer for the base, an integer for how many characters were consumed, and an error. This signals that it parses numerical literals and keeps track of format.
#Step 2: It first checks the initial rune must be a digit; otherwise, it immediately fails. This enforces that numbers must start correctly.
#Step 3: It initializes a helper struct (numberHelper) and begins scanning character by character.
#Step 4: When encountering non-digit characters, it switches on their type: a minus sign must appear only once and at the beginning; a dot indicates a decimal fraction; e or E mark scientific notation with proper handling of signs; b, o, or x specify binary, octal, or hexadecimal formats but only in correct positions.
#Step 5: The helper validates each of these cases, ensuring consistency in format. If anything is misplaced (like wrong base prefix, multiple minus signs, or invalid characters), the function raises a parse error.
#Step 6: Whitespace or newlines terminate the number cleanly, but any unexpected non-digit character inside the literal produces an error unless it is valid in hexadecimal context.
#Step 7: If parsing succeeds, the function returns the base of the number (from the helper), the number of runes read, and no error.
#Step 8: Summarizing, the function parses a numerical string, determines its base and length, and returns an error if the number is malformed."""

summarization_php_1 = """#Step 1: The method is named addOne, which hints at adding a single element to something.
#Step 2: It takes three parameters: a title, a URL, and an optional array of extra data. This suggests it is constructing some structured item.
#Step 3: Inside, it calls $this->addBreadcrumb(...), meaning the class maintains a breadcrumb collection.
#Step 4: The argument is BreadcrumbItem::make($title, $url, $data), which creates a breadcrumb item object from the provided values.
#Step 5: The method then returns the result of adding that item to the collection.
#Step 6: From this, the distilled meaning is: the function adds a breadcrumb item to the collection."""

summarization_php_2 = """#Step 1: The method is named order, which suggests it organizes items.
#Step 2: It retrieves the total count of breadcrumb items via $this->count().
#Step 3: It then maps over each breadcrumb item, resetting its position first.
#Step 4: The first item in the sequence is marked with setFirst(), and the last item is marked with setLast().
#Step 5: Each modified item is returned from the map, updating the collection.
#Step 6: Finally, the method returns $this, supporting method chaining.
Step 7: In summary, the function orders all breadcrumb items, resetting their positions and marking the first and last items appropriately."""

summarization_php_3 = """#Step 1: The method is named parse and takes a string and optional configuration options, indicating it processes text input.
#Step 2: It merges the provided options with defaults, setting up a configuration array and a flag for HTML purification.
#Step 3: The method calls parent::text($text), using the Parsedown Extra library to convert Markdown into HTML.
#Step 4: If HTML purification is enabled in the config and the purifier option is true, it obtains an HTMLPurifier instance.
#Step 5: The HTML content is then purified according to the given configuration, removing unsafe elements.
#Step 6: Finally, it returns the resulting sanitized HTML.
#Step 7: Summarizing, the function converts Markdown text to HTML and sanitizes the output."""

# ------------------------------Translation----------------------------
trans_java2cs_0 = """# Step 1: The method is named 'listSpeechSynthesisTasks' and follows Java's lowerCamelCase convention.
# Step 2: In C#, method names use PascalCase, so rename it to 'ListSpeechSynthesisTasks'.
# Step 3: The return type 'ListSpeechSynthesisTasksResult' in Java maps to 'ListSpeechSynthesisTasksResponse' in C# AWS SDK conventions.
# Step 4: The parameter type 'ListSpeechSynthesisTasksRequest' remains unchanged.
# Step 5: Java uses 'beforeClientExecution' for preprocessing the request, which is replaced in C# by creating an 'InvokeOptions' object.
# Step 6: Configure 'InvokeOptions' with the proper 'RequestMarshaller' and 'ResponseUnmarshaller' instances.
# Step 7: Replace 'executeListSpeechSynthesisTasks(request)' with 'Invoke<ListSpeechSynthesisTasksResponse>(request, options)'.
# Step 8: Add the 'public virtual' modifier in C# to follow SDK extensibility patterns.
# Step 9: The final summary: Convert Java’s pre-execution and execution pattern into C#’s InvokeOptions and Invoke call, adjusting naming conventions and type suffixes."""

trans_java2cs_1 = """# Step 1: The method is named 'updateJourneyState' and follows Java lowerCamelCase naming.
# Step 2: In C#, rename the method to 'UpdateJourneyState' to follow PascalCase naming.
# Step 3: Map the Java return type 'UpdateJourneyStateResult' to the C# convention 'UpdateJourneyStateResponse'.
# Step 4: Keep the parameter type 'UpdateJourneyStateRequest' unchanged but adjust formatting for C#.
# Step 5: Replace Java's 'request = beforeClientExecution(request);' with equivalent preprocessing in C# before calling the SDK invoke pattern.
# Step 6: Implement the C# SDK call by creating 'InvokeOptions', setting 'UpdateJourneyStateRequestMarshaller.Instance' and 'UpdateJourneyStateResponseUnmarshaller.Instance', then calling 'Invoke<UpdateJourneyStateResponse>(request, options)'.
# Step 7: Apply C# method modifiers (e.g., 'public virtual') and C# code style conventions.
# Step 8: The final summary: Map Java’s pre-execution + execute method into a C# InvokeOptions setup and Invoke<T> call, renaming the method and return type per C# conventions."""

trans_java2cs_2 = """# Step 1: The method is named 'removePresentationFormat' and follows Java lowerCamelCase naming.
# Step 2: In C#, rename the method to 'RemovePresentationFormat' to follow PascalCase naming.
# Step 3: The Java method calls a helper 'remove1stProperty(PropertyIDMap.PID_PRESFORMAT)'; identify whether an equivalent helper exists in C# or expand the helper inline.
# Step 4: If expanding inline in C#, obtain the first section as 'MutableSection s = (MutableSection)FirstSection;' then call 's.RemoveProperty(PropertyIDMap.PID_PRESFORMAT);'.
# Step 5: Keep the constant name 'PropertyIDMap.PID_PRESFORMAT' unchanged and ensure visibility/access semantics match C# usage.
# Step 6: Use C# naming for method calls (RemoveProperty) and follow C# casting and null-check idioms if necessary.
# Step 7: Apply C# formatting, method modifiers, and error handling consistent with the surrounding codebase.
# Step 8: The final summary: Rename the method, retain its logic by expanding the helper into explicit property removal using C# APIs."""

trans_cs2java_0 = """# Step 1: The method is named 'ListSpeechSynthesisTasks' and follows C# PascalCase naming.
# Step 2: In Java, rename the method to 'listSpeechSynthesisTasks' to follow lowerCamelCase naming.
# Step 3: Map the C# return type 'ListSpeechSynthesisTasksResponse' to the Java convention 'ListSpeechSynthesisTasksResult'.
# Step 4: Keep the parameter type 'ListSpeechSynthesisTasksRequest' unchanged but follow Java parameter formatting.
# Step 5: Replace C#'s explicit 'InvokeOptions' + marshaller/unmarshaller + 'Invoke<T>' with Java's pattern: call 'request = beforeClientExecution(request);' then call an 'executeListSpeechSynthesisTasks(request)' method that contains marshalling/unmarshalling and HTTP invocation.
# Step 6: Move marshaller/unmarshaller configuration into the Java 'execute...' method (encapsulate serialization logic rather than exposing InvokeOptions).
# Step 7: Use Java modifiers, exception handling, and brace style for the method body.
# Step 8: The final summary: Translate C#'s InvokeOptions/Invoke pattern into Java's beforeClientExecution + executeXxx encapsulation and change names and return-type suffixes accordingly."""

trans_cs2java_1 = """# Step 1: The method is named 'UpdateJourneyState' and follows C# PascalCase naming.
# Step 2: In Java, rename the method to 'updateJourneyState' to follow lowerCamelCase naming.
# Step 3: Map the C# return type 'UpdateJourneyStateResponse' to the Java convention 'UpdateJourneyStateResult'.
# Step 4: Keep the parameter type 'UpdateJourneyStateRequest' unchanged but use Java formatting and signature style.
# Step 5: Replace the C# 'InvokeOptions' + marshaller/unmarshaller + 'Invoke<...>' pattern with Java's 'request = beforeClientExecution(request);' followed by 'executeUpdateJourneyState(request)' that hides marshalling/unmarshalling details.
# Step 6: Ensure the Java 'executeUpdateJourneyState' method performs serialization, HTTP call, and response conversion analogous to C# unmarshaller behavior.
# Step 7: Use Java-specific modifiers, checked/unchecked exception handling (if applicable), and Java brace/indent style.
# Step 8: The final summary: Convert C#'s InvokeOptions+Invoke logic into Java's beforeClientExecution + executeXxx encapsulation and adapt naming and return suffixes for Java."""

trans_cs2java_2 = """# Step 1: The method is named 'RemovePresentationFormat' and follows C# PascalCase naming.
# Step 2: In Java, rename the method to 'removePresentationFormat' to follow lowerCamelCase naming.
# Step 3: The C# implementation casts and uses 'MutableSection s = (MutableSection)FirstSection; s.RemoveProperty(PropertyIDMap.PID_PRESFORMAT);' — plan to map that to Java semantics.
# Step 4: If Java has a helper 'remove1stProperty', use 'remove1stProperty(PropertyIDMap.PID_PRESFORMAT);' to keep the code concise.
# Step 5: Otherwise, translate the C# expansion into Java: obtain the first section (e.g., '(MutableSection) getFirstSection()') and call the Java-style instance method 's.removeProperty(PropertyIDMap.PID_PRESFORMAT);'.
# Step 6: Preserve the constant 'PropertyIDMap.PID_PRESFORMAT' and ensure method names follow Java conventions (removeProperty).
# Step 7: Apply Java formatting, access modifiers, and any needed null checks or exception handling.
# Step 8: The final summary: Convert C#'s explicit FirstSection cast and RemoveProperty call into either a Java helper call or an explicit cast plus s.removeProperty(...), and rename the method to Java naming."""

summarization_java = [summarization_java_1, summarization_java_2, summarization_java_3]
summarization_python = [summarization_python_1, summarization_python_2, summarization_python_3]
summarization['java'] = summarization_java
summarization['python'] = summarization_python
summarization['php'] = [summarization_php_1, summarization_php_2, summarization_php_3]
summarization['go'] = [summarization_go_1, summarization_go_2, summarization_go_3]
summarization['javascript'] = 0
summarization['ruby'] = 0

translation['java'] = [trans_java2cs_0, trans_java2cs_1, trans_java2cs_2]
translation['cs'] = [trans_cs2java_0, trans_cs2java_1, trans_cs2java_2]