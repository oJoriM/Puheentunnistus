README

requirements.txt
pip
miten asennetaan

## Yleiskuvaus

Tämä ohjelma mahdollistaa reaaliaikaisen puheohjauksen: käyttäjä
painaa määritettyä näppäintä, puhuu mikrofonille, ja ohjelma:

1.  Tallentaa äänen näppäimen ollessa painettuna
2.  Tunnistaa puheen Vosk-kirjaston avulla
3.  Lähettää tekstin OpenWebUI-mallille (esim. Gemma3)
4.  Toistaa vastauksen teksti--puheeksi (TTS) Coqui TTS:llä tai
    pyttsx3-varajärjestelmällä
5.  Animoi avatarin tunnetilan tai puhetilan mukaan

Ohjelma toimii paina-ja-puhu (PTT, *push-to-talk*) -periaatteella.

------------------------------------------------------------------------

## Käyttö

1.  Käynnistä ohjelma

2.  Pidä määritetty näppäin (`PTT_KEY`, oletuksena `'`) pohjassa
    puhuaksesi.

3.  Ohjelma:

    -   Nauhoittaa puheen
    -   Tunnistaa sen tekstiksi
    -   Lähettää mallille
    -   Näyttää avatarin reaktiot


------------------------------------------------------------------------

## Ympäristömuuttujat

  Muuttuja               Kuvaus
  ---------------------- ------------------------------------
  `PTT_KEY`              Näppäin, joka aktivoi nauhoituksen
  `API_KEY`              (Valinnainen) OpenWebUI API-avain
  `OPENWEBUI_BASE_URL`   Mallipalvelimen osoite
  `VOSK_MODEL_DIR`       Vosk-mallin hakemisto

------------------------------------------------------------------------

## Kirjastot

  Kirjasto                        Tarkoitus
  ------------------------------- -------------------------------
  `sounddevice`                   Mikrofonin äänisyöte
  `vosk`                          Puheentunnistus
  `requests`                      API-kutsut OpenWebUI-mallille
  `TTS (Coqui)`                   Tekstin muuttaminen puheeksi
  `pyttsx3`                       TTS-varajärjestelmä
  `simpleaudio` / `sounddevice`   Äänen toisto
  `pynput`                        Näppäimistön kuuntelu (PTT)
  `pygame`                        Avatar-ikkuna ja animaatiot
  `numpy`                         Äänidatan käsittely

------------------------------------------------------------------------

## Rakenteen pääkohdat

-   **ASR (puheentunnistus):** Vosk-malli ja mikrofonin tallennus
-   **TTS:** Coqui TTS ensisijaisena, pyttsx3 varajärjestelmänä
-   **Avatar:** pygame-animaatiot tunnetilan mukaan
-   **PTT-kuuntelu:** pynput
-   **Mallikutsu:** OpenWebUI / LLM-rajapinta

------------------------------------------------------------------------

## Avatarin tunnetilat

Avatar reagoi tekstistä tunnistettuihin sanoihin (esim. *happy, sad,
angry, scared, yes, no*) ja muuttaa kuvan sekä tilatekstin vastaavasti.

------------------------------------------------------------------------
