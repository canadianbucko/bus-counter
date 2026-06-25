package main

import (
	"bufio"
	"crypto/rand"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	Addr         string
	DataDir      string
	ModelPath    string
	TargetClass  string
	Confidence   string
	PythonBin    string
	DetectorPath string
}

type DetectionObject struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	X1         float64 `json:"x1"`
	Y1         float64 `json:"y1"`
	X2         float64 `json:"x2"`
	Y2         float64 `json:"y2"`
}

type DetectorResponse struct {
	Count     int               `json:"count"`
	Objects   []DetectionObject `json:"objects"`
	ElapsedMS int64             `json:"elapsed_ms"`
	Model     string            `json:"model"`
	Target    string            `json:"target"`
}

type HistoryRecord struct {
	ID           string            `json:"id"`
	CreatedAt    string            `json:"created_at"`
	OriginalName string            `json:"original_name"`
	UploadPath   string            `json:"upload_path"`
	OutputPath   string            `json:"output_path"`
	BusCount     int               `json:"bus_count"`
	Objects      []DetectionObject `json:"objects"`
	ElapsedMS    int64             `json:"elapsed_ms"`
	Model        string            `json:"model"`
	Target       string            `json:"target"`
}

type App struct {
	cfg       Config
	historyMu sync.Mutex
}

func main() {
	cfg := Config{
		Addr:         env("HTTP_ADDR", ":8080"),
		DataDir:      env("DATA_DIR", "data"),
		ModelPath:    env("MODEL_PATH", "yolov8n.pt"),
		TargetClass:  env("TARGET_CLASS", "bus"),
		Confidence:   env("CONFIDENCE", "0.25"),
		PythonBin:    env("PYTHON_BIN", "python3"),
		DetectorPath: env("DETECTOR_PATH", "detector/detect.py"),
	}

	app := &App{cfg: cfg}
	if err := app.ensureDirs(); err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", app.handleIndex)
	mux.HandleFunc("POST /api/v1/detect", app.handleDetect)
	mux.HandleFunc("GET /api/v1/history", app.handleHistory)
	mux.HandleFunc("DELETE /api/v1/history", app.handleClearHistory)
	mux.HandleFunc("GET /api/v1/reports/csv", app.handleCSVReport)
	mux.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.Dir("web/assets"))))
	mux.Handle("/outputs/", http.StripPrefix("/outputs/", http.FileServer(http.Dir(filepath.Join(cfg.DataDir, "outputs")))))
	mux.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir(filepath.Join(cfg.DataDir, "uploads")))))

	handler := logging(cors(mux))
	log.Printf("server started on %s", cfg.Addr)
	log.Printf("http://localhost%s", strings.TrimPrefix(cfg.Addr, ":"))
	if err := http.ListenAndServe(cfg.Addr, handler); err != nil {
		log.Fatal(err)
	}
}

func (a *App) handleIndex(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, filepath.Join("web", "index.html"))
}

func (a *App) handleDetect(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "не удалось прочитать форму: "+err.Error())
		return
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, "нужно загрузить файл в поле image")
		return
	}
	defer file.Close()

	id := newID()
	ext := safeExt(header.Filename)
	if ext == "" {
		ext = ".jpg"
	}
	uploadRel := filepath.ToSlash(filepath.Join("uploads", id+ext))
	outputRel := filepath.ToSlash(filepath.Join("outputs", id+"_result.jpg"))
	uploadAbs := filepath.Join(a.cfg.DataDir, uploadRel)
	outputAbs := filepath.Join(a.cfg.DataDir, outputRel)

	if err := saveUploadedFile(file, uploadAbs); err != nil {
		writeError(w, http.StatusInternalServerError, "не удалось сохранить файл: "+err.Error())
		return
	}

	det, raw, err := a.runDetector(uploadAbs, outputAbs)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("ошибка детектора: %v; вывод: %s", err, raw))
		return
	}

	record := HistoryRecord{
		ID:           id,
		CreatedAt:    time.Now().Format(time.RFC3339),
		OriginalName: sanitizeName(header.Filename),
		UploadPath:   "/" + uploadRel,
		OutputPath:   "/" + outputRel,
		BusCount:     det.Count,
		Objects:      det.Objects,
		ElapsedMS:    det.ElapsedMS,
		Model:        det.Model,
		Target:       det.Target,
	}
	if record.Model == "" {
		record.Model = a.cfg.ModelPath
	}
	if record.Target == "" {
		record.Target = a.cfg.TargetClass
	}

	if err := a.appendHistory(record); err != nil {
		writeError(w, http.StatusInternalServerError, "не удалось сохранить историю: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, record)
}

func (a *App) runDetector(input, output string) (DetectorResponse, string, error) {
	args := []string{
		a.cfg.DetectorPath,
		"--input", input,
		"--output", output,
		"--model", a.cfg.ModelPath,
		"--target", a.cfg.TargetClass,
		"--conf", a.cfg.Confidence,
	}
	cmd := exec.Command(a.cfg.PythonBin, args...)
	cmd.Env = os.Environ()

	out, err := cmd.Output()
	if err != nil {
		return DetectorResponse{}, strings.TrimSpace(string(out)), err
	}

	var det DetectorResponse
	text := strings.TrimSpace(string(out))
	if err := json.Unmarshal([]byte(text), &det); err != nil {
		return DetectorResponse{}, text, fmt.Errorf("не удалось прочитать ответ детектора: %w", err)
	}
	return det, text, nil
}

func (a *App) handleHistory(w http.ResponseWriter, r *http.Request) {
	records, err := a.readHistory()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, records)
}

func (a *App) handleClearHistory(w http.ResponseWriter, r *http.Request) {
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	path := filepath.Join(a.cfg.DataDir, "history.jsonl")
	if err := os.WriteFile(path, nil, 0644); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (a *App) handleCSVReport(w http.ResponseWriter, r *http.Request) {
	records, err := a.readHistory()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	name := "bus_counter_report_" + time.Now().Format("20060102_150405") + ".csv"
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	_, _ = w.Write([]byte{0xEF, 0xBB, 0xBF})

	cw := csv.NewWriter(w)
	_ = cw.Write([]string{"ID", "Дата", "Исходный файл", "Количество автобусов", "Время обработки, мс", "Модель", "Результат"})
	for _, rec := range records {
		_ = cw.Write([]string{rec.ID, rec.CreatedAt, rec.OriginalName, strconv.Itoa(rec.BusCount), strconv.FormatInt(rec.ElapsedMS, 10), rec.Model, rec.OutputPath})
	}
	cw.Flush()
}

func (a *App) appendHistory(record HistoryRecord) error {
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	path := filepath.Join(a.cfg.DataDir, "history.jsonl")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(record)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	return err
}

func (a *App) readHistory() ([]HistoryRecord, error) {
	a.historyMu.Lock()
	defer a.historyMu.Unlock()
	path := filepath.Join(a.cfg.DataDir, "history.jsonl")
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return []HistoryRecord{}, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var records []HistoryRecord
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" {
			continue
		}
		var rec HistoryRecord
		if err := json.Unmarshal([]byte(line), &rec); err == nil {
			records = append(records, rec)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	sort.Slice(records, func(i, j int) bool { return records[i].CreatedAt > records[j].CreatedAt })
	return records, nil
}

func (a *App) ensureDirs() error {
	for _, dir := range []string{"uploads", "outputs", "reports"} {
		if err := os.MkdirAll(filepath.Join(a.cfg.DataDir, dir), 0755); err != nil {
			return err
		}
	}
	path := filepath.Join(a.cfg.DataDir, "history.jsonl")
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return os.WriteFile(path, nil, 0644)
	}
	return nil
}

func saveUploadedFile(src multipart.File, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, src)
	return err
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}

func env(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func newID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b)
}

func safeExt(name string) string {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".webp", ".bmp":
		return ext
	default:
		return ""
	}
}

func sanitizeName(name string) string {
	name = filepath.Base(name)
	name = strings.ReplaceAll(name, "\x00", "")
	if name == "." || name == string(filepath.Separator) || strings.TrimSpace(name) == "" {
		return "image"
	}
	return name
}
