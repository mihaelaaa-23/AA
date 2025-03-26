document.addEventListener("DOMContentLoaded", function () {
const algorithmSelect = document.getElementById("algorithm");
const dataTypeSelect = document.getElementById("dataType");
const arraySizeInput = document.getElementById("arraySize");
const generateArrayButton = document.getElementById("generateArray");
const startSortButton = document.getElementById("startSort");
const arrayBarsContainer = document.getElementById("arrayBars");

document.getElementById("pauseSort").addEventListener("click", () => {
    if (isSorting) {
        if (!isPaused) {
            isPaused = true;
            document.getElementById("pauseSort").innerText = "Resume";
        } else {
            isPaused = false;
            document.getElementById("pauseSort").innerText = "Pause/Stop";
        }
    } else {
        stopSorting = true; // **Fully stop sorting**
        isSorting = false;
    }
});

let array = [];
let isSorting = false;
let isPaused = false;
let stopSorting = false;

// Generate random array based on selected options
function generateArray() {
    const size = parseInt(arraySizeInput.value);
    const dataType = dataTypeSelect.value;
    const minValue = parseInt(document.getElementById("minValue").value) || 1;
    const maxValue = parseInt(document.getElementById("maxValue").value) || 100;

    array = [];
    for (let i = 0; i < size; i++) {
        switch (dataType) {
            case "randomInt":
                array.push(Math.floor(Math.random() * (maxValue - minValue + 1)) + minValue);
                break;
            case "randomFloat":
                array.push(parseFloat(((Math.random() * (maxValue - minValue)) + minValue).toFixed(2)));
                break;
            case "negative":
                array.push(Math.floor(Math.random() * (maxValue - minValue + 1)) + minValue - (maxValue - minValue) / 2);
                break;
            case "halfSorted":
                array.push(Math.floor(Math.random() * (maxValue - minValue + 1)) + minValue);
                if (i === Math.floor(size / 2)) array.sort((a, b) => a - b);
                break;
            case "sorted":
                array.push(Math.floor(Math.random() * (maxValue - minValue + 1)) + minValue);
                array.sort((a, b) => a - b);
                break;
            case "descending":
                array.push(Math.floor(Math.random() * (maxValue - minValue + 1)) + minValue);
                array.sort((a, b) => b - a);
                break;
        }
    }

    operations = 0;
    updateOperationsCount();
    console.log("Generated array:", array); // Debugging log
    renderArray(); // Ensure array is rendered
}


  function updateOperationsCount() {
    document.getElementById("operationsCount").innerHTML = `
        Comparisons: ${comparisons} | Swaps: ${swaps} | Recursion Depth: ${recursionDepth}
    `;
    }


// Render array as bars
function renderArray(highlightIndices = []) {
    arrayBarsContainer.innerHTML = "";
    const size = array.length;
    const containerWidth = arrayBarsContainer.clientWidth; // Get container width
    const maxBarWidth = 40;
    const minBarWidth = 3;
    const spacing = 2;
    
    // Ensure bars fit dynamically
    let barWidth = Math.max(minBarWidth, Math.min(maxBarWidth, (containerWidth - spacing * size) / size));
    
    arrayBarsContainer.style.width = `${Math.max(100, barWidth * size + spacing * size)}px`; // Expand if needed

    array.forEach((value, index) => {
        const barContainer = document.createElement("div");
        barContainer.style.display = "flex";
        barContainer.style.flexDirection = "column";
        barContainer.style.alignItems = "center";
        barContainer.style.margin = `0 ${spacing / 2}px`;

        const barLabel = document.createElement("span");
        barLabel.innerText = value;
        barLabel.style.color = "#000";
        barLabel.style.fontSize = "12px";
        barLabel.style.marginBottom = "2px";

        // Hide numbers if bars are too narrow
        if (barWidth < 12) {
            barLabel.style.display = "none";
        }

        const bar = document.createElement("div");
        bar.classList.add("bar");
        const maxHeight = 300; // Fixed height for visualization area
        const maxValueInArray = Math.max(...array); // Get highest value in array
        const scaleFactor = maxHeight / maxValueInArray; // Scale bars dynamically

        bar.style.height = `${value * scaleFactor}px`; // Apply scaling

        bar.style.width = `${barWidth}px`;

        if (highlightIndices.includes(index)) {
            bar.style.backgroundColor = "rgb(255, 0, 255)"; // Magenta for swaps
        } else {
            bar.style.backgroundColor = "#00BFFF"; // Blue for normal bars
        }

        barContainer.appendChild(barLabel);
        barContainer.appendChild(bar);
        arrayBarsContainer.appendChild(barContainer);
    });

    updateOperationsCount();
}


// Sorting algorithms with visualization
let comparisons = 0;
let swaps = 0;
let recursionDepth = 0;

async function quickSort(arr, low, high, depth = 0) {
    if (low < high) {
        recursionDepth = Math.max(recursionDepth, depth); // Track max recursion depth
        const pi = await partition(arr, low, high);
        await quickSort(arr, low, pi - 1, depth + 1);
        await quickSort(arr, pi + 1, high, depth + 1);
    }
}

async function partition(arr, low, high) {
    const pivot = arr[high];
    let i = low - 1;
    for (let j = low; j < high; j++) {
        comparisons++; // Counting comparisons
        if (arr[j] < pivot) {
            i++;
            if (i !== j) { // **Only count swaps if indices are different**
                [arr[i], arr[j]] = [arr[j], arr[i]];
                swaps++; 
            }
            renderArray([i, j]);
            await sleep(100);
            if (stopSorting) return; // Stop execution immediately
            while (isPaused) await sleep(100); // Pause execution
        }
    }
    if (i + 1 !== high) { // **Only count swap if necessary**
        [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
        swaps++;
    }
    renderArray([i + 1, high]);
    await sleep(100);
    return i + 1;
}



async function mergeSort(arr, left = 0, right = arr.length - 1, depth = 0) {
    if (left >= right) return;

    recursionDepth = Math.max(recursionDepth, depth);
    const mid = Math.floor((left + right) / 2);
    await mergeSort(arr, left, mid, depth + 1);
    await mergeSort(arr, mid + 1, right, depth + 1);
    await merge(arr, left, mid, right);
}

async function merge(arr, left, mid, right) {
    let leftArr = arr.slice(left, mid + 1);
    let rightArr = arr.slice(mid + 1, right + 1);
    let i = 0, j = 0, k = left;

    while (i < leftArr.length && j < rightArr.length) {
        comparisons++; // Count comparisons
        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
        renderArray([k]);
        await sleep(100);
        if (stopSorting) return;
        while (isPaused) await sleep(100);
    }
    
    while (i < leftArr.length) {
        arr[k++] = leftArr[i++];
    }

    while (j < rightArr.length) {
        arr[k++] = rightArr[j++];
    }
}

async function heapSort(arr) {
    for (let i = Math.floor(arr.length / 2) - 1; i >= 0; i--) {
        await heapify(arr, arr.length, i);
    }

    for (let i = arr.length - 1; i > 0; i--) {
        if (stopSorting) return; // Stop execution immediately
        while (isPaused) await sleep(100); // Pause execution

        [arr[0], arr[i]] = [arr[i], arr[0]];
        swaps++;
        renderArray([0, i]);
        await sleep(100);
        await heapify(arr, i, 0);
    }
}

async function heapify(arr, n, i) {
    let largest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;

    if (left < n) {
        comparisons++;
        if (arr[left] > arr[largest]) largest = left;
    }

    if (right < n) {
        comparisons++;
        if (arr[right] > arr[largest]) largest = right;
    }

    if (largest !== i) {
        [arr[i], arr[largest]] = [arr[largest], arr[i]];
        swaps++; // **Only count actual swaps**
        renderArray([i, largest]);
        await sleep(100);
        await heapify(arr, n, largest);
    }
}

async function bubbleSort(arr) {
    let n = arr.length;
    let swapped;

    for (let i = 0; i < n - 1; i++) {
        swapped = false; // Reset swap flag
        if (stopSorting) return; // Stop execution immediately
        while (isPaused) await sleep(100); // Pause execution

        for (let j = 0; j < n - i - 1; j++) {
            comparisons++; // Count comparisons
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
                swaps++; // Count swaps
                swapped = true;
                renderArray([j, j + 1]);
                await sleep(100);
            }
        }

        if (!swapped) break; // **Exit early if no swaps occurred**
    }
    isSorting = false;
}


// Utility function to delay execution
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Event listeners
generateArrayButton.addEventListener("click", generateArray);
startSortButton.addEventListener("click", async () => {
    comparisons = 0;
    swaps = 0;
    recursionDepth = 0;
    stopSorting = false; // **Reset stop state**
    updateOperationsCount();

    const algorithm = algorithmSelect.value;
    if (algorithm === "quick") {
        await quickSort(array, 0, array.length - 1);
    } else if (algorithm === "merge") {
        await mergeSort(array, 0, array.length - 1);
    } else if (algorithm === "heap") {
        await heapSort(array);
    } else if (algorithm === "bubble") {
        await bubbleSort(array);
    }

    updateOperationsCount();
    renderArray();
});

// Initial array generation
generateArray();
});